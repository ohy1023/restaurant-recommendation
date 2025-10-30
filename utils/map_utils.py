import folium


def create_restaurant_map(center_location, user_location, restaurants_df):
    """
    음식점 위치를 표시하는 지도 생성

    Args:
        center_location: (latitude, longitude) - 지도 중심
        user_location: (latitude, longitude) - 사용자 위치
        restaurants_df: 음식점 정보 DataFrame (Store, lat, lng, Score, url 컬럼 필요)

    Returns:
        folium.Map 객체
    """
    # 지도 생성
    m = folium.Map(
        location=center_location,
        zoom_start=15
    )

    # 사용자 위치 마커
    folium.Marker(
        user_location,
        tooltip='현위치',
        icon=folium.Icon(color='blue', icon='user', prefix='fa')
    ).add_to(m)

    # 음식점 마커
    for idx in range(len(restaurants_df)):
        restaurant = restaurants_df.iloc[idx]

        popup_html = f"""
        <div style="width: 200px;">
            <h4>{restaurant['Store']}</h4>
            <p><b>총점:</b> {restaurant['Total Score']}</p>
            <p><b>리뷰 수:</b> {restaurant['Review']}</p>
            <a href="{restaurant['url']}" target="_blank">상세 보기</a>
        </div>
        """

        folium.Marker(
            [restaurant['lat'], restaurant['lng']],
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=restaurant['Store'],
            icon=folium.Icon(color='red', icon='cutlery', prefix='fa')
        ).add_to(m)

    return m


def create_simple_map(location, zoom_start=15):
    """
    단순 지도 생성

    Args:
        location: (latitude, longitude)
        zoom_start: 초기 줌 레벨

    Returns:
        folium.Map 객체
    """
    return folium.Map(
        location=location,
        zoom_start=zoom_start
    )


def add_marker(map_obj, location, popup_text=None, tooltip_text=None,
               color='red', icon='info-sign'):
    """
    지도에 마커 추가

    Args:
        map_obj: folium.Map 객체
        location: (latitude, longitude)
        popup_text: 클릭 시 표시될 텍스트
        tooltip_text: 마우스 오버 시 표시될 텍스트
        color: 마커 색상
        icon: 아이콘 이름

    Returns:
        map_obj: 마커가 추가된 지도
    """
    folium.Marker(
        location,
        popup=popup_text,
        tooltip=tooltip_text,
        icon=folium.Icon(color=color, icon=icon)
    ).add_to(map_obj)

    return map_obj


def add_route_line(map_obj, coordinates, color='blue', weight=5, opacity=0.7):
    """
    지도에 경로 선 추가

    Args:
        map_obj: folium.Map 객체
        coordinates: [(lat, lng), ...] 좌표 리스트
        color: 선 색상
        weight: 선 두께
        opacity: 투명도

    Returns:
        map_obj: 경로가 추가된 지도
    """
    folium.PolyLine(
        coordinates,
        color=color,
        weight=weight,
        opacity=opacity
    ).add_to(map_obj)

    return map_obj


def add_circle(map_obj, location, radius, color='blue', fill=True,
               fill_opacity=0.2, popup_text=None):
    """
    지도에 원 추가

    Args:
        map_obj: folium.Map 객체
        location: (latitude, longitude) - 원의 중심
        radius: 반지름 (미터)
        color: 원 색상
        fill: 내부 채우기 여부
        fill_opacity: 채우기 투명도
        popup_text: 클릭 시 표시될 텍스트

    Returns:
        map_obj: 원이 추가된 지도
    """
    folium.Circle(
        location,
        radius=radius,
        color=color,
        fill=fill,
        fill_opacity=fill_opacity,
        popup=popup_text
    ).add_to(map_obj)

    return map_obj