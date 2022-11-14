from django.db.models import Prefetch
from django.test import TestCase

from .models import restaurant_review, restaurant_info


# Create your tests here.
class YourTestClass(TestCase):

    @classmethod
    def setUpTestData(cls):
        restaurant_info.objects.create(id=1, name='test', x=124.34, y=36.64,
                                       address='서울 영등포구 도림로 54길', url='www.naver.com',
                                       type='피자')
        restaurant_info.objects.create(id=2, name='test', x=124.34, y=36.64,
                                       address='서울 영등포구 도림로 54길', url='www.naver.com',
                                       type='치킨')
        restaurant_review.objects.create(id=1, score=3, review='사장님은 예쁜데 싸가지가 존나 없어요.',
                                         restaurant_id=restaurant_info.objects.get(id=1))
        restaurant_review.objects.create(id=2, score=4, review='진짜 개맛있음요.',
                                         restaurant_id=restaurant_info.objects.get(id=1))
        restaurant_review.objects.create(id=3, score=1, review='개노맛.',
                                         restaurant_id=restaurant_info.objects.get(id=2))

    def test_select_type(self):
        query = restaurant_info.objects.get(id=1)

        self.assertEquals(query.type,'피자')

    def test_select_review(self):
        query = restaurant_review.objects.get(id=1)

        self.assertEquals(query.review,'사장님은 예쁜데 싸가지가 존나 없어요.')

    def test_select_join(self):
        query_set = restaurant_review.objects.filter(restaurant_id=1,restaurant_id__id=1).select_related('restaurant_id').prefetch_related('restaurant_id__restaurant_review_set')

        result = [{
            "restaurant_id": review.restaurant_id.id,
            "name":review.restaurant_id.name,
            "x":review.restaurant_id.x,
            "y": review.restaurant_id.y,
            "address": review.restaurant_id.address,
            "url": review.restaurant_id.url,
            "id":review.id,
            "score":review.score,
            "review":review.review
        }for review in query_set]
        self.assertEquals(len(result),2)
