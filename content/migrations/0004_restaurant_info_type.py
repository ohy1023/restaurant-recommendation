# Generated by Django 3.2.16 on 2022-11-12 12:16

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('content', '0003_alter_restaurant_info_id'),
    ]

    operations = [
        migrations.AddField(
            model_name='restaurant_info',
            name='type',
            field=models.CharField(default='', max_length=30),
        ),
    ]