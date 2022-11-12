from django.db import models

# Create your models here.


class restaurant_info(models.Model):
    id = models.IntegerField(primary_key=True)
    name = models.CharField(max_length=50)
    x = models.FloatField()
    y = models.FloatField()
    address = models.CharField(max_length=100)
    url = models.CharField(max_length=100)
    type = models.CharField(max_length=30, default='')