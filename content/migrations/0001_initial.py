# Generated by Django 3.2.16 on 2022-11-12 07:01

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='MainappModel',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('main_intro', models.CharField(blank=True, max_length=255, null=True, verbose_name='메인 소개글')),
            ],
        ),
    ]
