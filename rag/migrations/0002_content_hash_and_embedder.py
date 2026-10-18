from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("rag", "0001_initial")]

    operations = [
        migrations.AddField(
            model_name="document",
            name="content_hash",
            field=models.CharField(blank=True, max_length=64, null=True, unique=True),
        ),
        migrations.AddField(
            model_name="chunk",
            name="embedder",
            field=models.CharField(db_index=True, default="hashing", max_length=120),
        ),
    ]
