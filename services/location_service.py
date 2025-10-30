import requests
from config.settings import GOOGLE_API_KEY


def get_current_location_google():
    """
    Google Geolocation API를 사용하여 현재 위치 가져오기

    Returns:
        (latitude, longitude) 튜플
    """
    url = f'https://www.googleapis.com/geolocation/v1/geolocate?key={GOOGLE_API_KEY}'
    data = {'considerIp': True}

    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        location = response.json()['location']
        lat = location['lat']
        lng = location['lng']
        return lat, lng
    except Exception as e:
        print(f"위치 가져오기 실패: {e}")
        return None, None

def get_current_location_ip():
    """
    IP 기반으로 사용자의 위치 가져오기

    Returns:
        (latitude, longitude) 튜플
    """
    try:
        res = requests.get("https://ipinfo.io")
        res.raise_for_status()
        data = res.json()
        loc = data.get('loc')  # "위도,경도" 형식의 문자열
        if loc:
            lat_str, lng_str = loc.split(',')
            return float(lat_str), float(lng_str)
        else:
            return None, None
    except Exception as e:
        print(f"IP 기반 위치 가져오기 실패: {e}")
        return None, None
