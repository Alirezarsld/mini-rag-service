from dataclasses import asdict
from pathlib import Path

from rest_framework import status
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from . import selectors, services
from .permissions import HasApiKey
from .serializers import DocumentCreateSerializer, QuerySerializer, document_to_dict


MAX_UPLOAD_BYTES = 1_000_000


def _index(title: str, text: str, source: str = "") -> Response:
    """Shared by the JSON and file-upload endpoints so both handle errors identically."""
    try:
        document = services.ingest_document(title=title[:200], text=text, source=source)
    except services.DuplicateDocument as dup:
        return Response({"detail": "This text is already indexed.",
                         "document_id": dup.document.id}, status=status.HTTP_409_CONFLICT)
    except ValueError as exc:
        return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
    return Response(document_to_dict(selectors.get_document(document.id)),
                    status=status.HTTP_201_CREATED)


class HealthView(APIView):
    permission_classes = []
    throttle_classes = []

    def get(self, request):
        return Response({"status": "ok"})


class DocumentListCreateView(APIView):
    permission_classes = [HasApiKey]

    def get(self, request):
        return Response([document_to_dict(d) for d in selectors.list_documents()])

    def post(self, request):
        serializer = DocumentCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        return _index(**serializer.validated_data)


class DocumentUploadView(APIView):
    """``multipart/form-data`` with a UTF-8 ``file`` (.md/.txt) and an optional ``title``."""

    permission_classes = [HasApiKey]
    parser_classes = [MultiPartParser]

    def post(self, request):
        upload = request.FILES.get("file")
        if upload is None:
            return Response({"detail": "Send the text file in a 'file' field."},
                            status=status.HTTP_400_BAD_REQUEST)
        if upload.size > MAX_UPLOAD_BYTES:
            return Response({"detail": "File is larger than 1 MB."},
                            status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)
        try:
            text = upload.read().decode("utf-8")
        except UnicodeDecodeError:
            return Response({"detail": "File must be UTF-8 text."},
                            status=status.HTTP_400_BAD_REQUEST)
        title = request.data.get("title") or Path(upload.name).stem
        return _index(title, text, source=upload.name)


class DocumentDetailView(APIView):
    permission_classes = [HasApiKey]

    def get(self, request, pk):
        document = selectors.get_document(pk)
        if document is None:
            return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)
        return Response(document_to_dict(document))

    def delete(self, request, pk):
        document = selectors.get_document(pk)
        if document is None:
            return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)
        services.delete_document(document)
        return Response(status=status.HTTP_204_NO_CONTENT)


class SearchView(APIView):
    permission_classes = [HasApiKey]

    def post(self, request):
        serializer = QuerySerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        hits = services.search(data["query"], data["top_k"])
        return Response({"results": [asdict(h) for h in hits]})


class AskView(APIView):
    permission_classes = [HasApiKey]

    def post(self, request):
        serializer = QuerySerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        return Response(services.answer_question(serializer.validated_data["query"],
                                                 serializer.validated_data["top_k"]))
