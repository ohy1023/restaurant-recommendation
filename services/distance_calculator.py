import osmnx as ox
import networkx as nx
from tqdm import tqdm


class DistanceCalculator:
    """OSMnx를 이용한 거리 계산 클래스"""

    def __init__(self, center_point, network_type='drive', dist=5500):
        """
        Args:
            center_point: (latitude, longitude) 튜플
            network_type: 네트워크 타입 ('drive', 'walk', 'bike' 등)
            dist: 중심점으로부터의 거리 (미터)
        """
        ox.config(use_cache=True, log_console=False)
        self.center_point = center_point
        self.graph = ox.graph_from_point(
            center_point,
            network_type=network_type,
            dist=dist
        )

    def get_shortest_path(self, origin, destination):
        """
        두 지점 사이의 최단 경로 계산

        Args:
            origin: (latitude, longitude)
            destination: (latitude, longitude)

        Returns:
            route: 경로 노드 리스트
        """
        orig_node = ox.get_nearest_node(self.graph, origin)
        dest_node = ox.get_nearest_node(self.graph, destination)
        route = nx.shortest_path(self.graph, orig_node, dest_node, weight='length')
        return route

    def get_distance(self, origin, destination):
        """
        두 지점 사이의 최단 거리 계산 (킬로미터)

        Args:
            origin: (latitude, longitude)
            destination: (latitude, longitude)

        Returns:
            distance: 거리 (km)
        """
        try:
            orig_node = ox.get_nearest_node(self.graph, origin)
            dest_node = ox.get_nearest_node(self.graph, destination)
            distance = nx.shortest_path_length(
                self.graph,
                orig_node,
                dest_node,
                weight='length'
            ) / 1000
            return round(distance, 2)
        except:
            return None

    def calculate_distances_to_restaurants(self, origin, restaurants_df):
        """
        사용자 위치에서 여러 음식점까지의 거리 계산

        Args:
            origin: (latitude, longitude)
            restaurants_df: 음식점 정보 DataFrame (pk, X, Y 컬럼 필요)

        Returns:
            distances: 거리 리스트
        """
        distances = []

        for idx in tqdm(range(len(restaurants_df)), desc="거리 계산 중"):
            rest_x = restaurants_df['X'].iloc[idx]
            rest_y = restaurants_df['Y'].iloc[idx]

            # 같은 위치면 스킵
            if (rest_x == origin[1]) and (rest_y == origin[0]):
                distances.append(0)
                continue

            distance = self.get_distance(origin, (rest_y, rest_x))
            distances.append(distance if distance is not None else 999.9)

        return distances

    def plot_route(self, origin, destination):
        """
        경로를 지도에 표시

        Args:
            origin: (latitude, longitude)
            destination: (latitude, longitude)

        Returns:
            fig, ax: matplotlib figure와 axes
        """
        route = self.get_shortest_path(origin, destination)
        fig, ax = ox.plot_graph_route(self.graph, route, node_size=0)
        return fig, ax

    def plot_graph(self):
        """그래프 시각화"""
        fig, ax = ox.plot_graph(
            self.graph,
            node_size=0,
            edge_linewidth=0.5
        )
        return fig, ax