# Generated by Django 3.2.16 on 2022-11-13 10:27

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('content', '0005_restaurant_review'),
    ]

    operations = [
        migrations.AlterField(
            model_name='restaurant_review',
            name='review',
            field=models.TextField(),
        ),
    ]