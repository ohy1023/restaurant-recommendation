# Generated by Django 3.2.16 on 2022-11-25 10:50

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('content', '0013_auto_20221125_1936'),
    ]

    operations = [
        migrations.AlterField(
            model_name='bad_word',
            name='word',
            field=models.CharField(max_length=100),
        ),
        migrations.AlterField(
            model_name='good_word',
            name='word',
            field=models.CharField(max_length=100),
        ),
    ]
