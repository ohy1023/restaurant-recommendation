# 1번 파일이 실행될 때 환경변수에 현재 자신의 프로젝트의 settings.py파일 경로를 등록.
import os

import pandas as pd

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "djangoProject4.settings")

# 2번 실행파일에 Django 환경을 불러오는 작업.
import django

django.setup()

# 3번 크롤링을 하고 DB model에 저장.
from content.models import restaurant_review, restaurant_info


def read_csv():
    df = pd.read_csv('reviews.csv', encoding='utf8')
    df = df.drop(['Unnamed: 0'], axis=1)

    return df


if __name__ == '__main__':
    for i in read_csv().index:
        restaurant_review(restaurant_id=restaurant_info.objects.get(id=read_csv()['restaurant_id'][i]), score=read_csv()['score'][i]
                          , review=read_csv()['review'][i]).save()