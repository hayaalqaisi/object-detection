"""
This file tells the database how to build the table for our Prediction model.
Django can also generate it for you with:  python manage.py makemigrations
It is included here so the project runs after just `migrate`.
"""

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="Prediction",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True,
                                           serialize=False, verbose_name="ID")),
                ("image", models.ImageField(upload_to="uploads/")),
                ("label", models.CharField(blank=True, max_length=20)),
                ("confidence", models.FloatField(default=0.0)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
        ),
    ]