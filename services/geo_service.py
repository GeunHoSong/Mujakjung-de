
import math


def calculate_distance(lat1, lon1, lat2, lon2):
    """
    두 위도 / 경도 좌표 간의 거리를 계산 (Haversine 공식)
    반환값: km
    """
    R = 6371.0

    lat1_rad = math.radians(float(lat1))
    lon1_rad = math.radians(float(lon1))
    lat2_rad = math.radians(float(lat2))
    lon2_rad = math.radians(float(lon2))

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad)
        * math.cos(lat2_rad)
        * math.sin(dlon / 2) ** 2
    )

    # 부동소수점 오차로 a가 1을 약간 넘는 경우 방지
    a = max(0.0, min(1.0, a))

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def optimize_route(places, start_lat=None, start_lng=None):
    """
    좌표가 있는 장소를 가까운 순서대로 정렬한다.
    좌표가 없는 장소는 마지막에 배치한다.
    현재는 첫 번째 장소부터 경로를 계산한다.
    """
    if not places:
        return []

    places_with_coords = []
    places_without_coords = []

    # 1. 좌표가 있는 장소와 없는 장소 분리
    for place in places:
        lat = place.get("y")
        lng = place.get("x")

        if lat is not None and lng is not None:
            try:
                lat = float(lat)
                lng = float(lng)

                # 위도와 경도의 유효 범위 확인
                if -90 <= lat <= 90 and -180 <= lng <= 180:
                    places_with_coords.append(place)
                else:
                    places_without_coords.append(place)

            except (TypeError, ValueError):
                places_without_coords.append(place)
        else:
            places_without_coords.append(place)

    # 2. 좌표가 있는 장소가 없으면 원래 순서로 반환
    if not places_with_coords:
        return places_without_coords

    unvisited = places_with_coords.copy()
    optimized_route = []

    # 3. 첫 번째 장소를 출발점으로 설정
    if start_lat is not None and start_lng is not None:
        start_lat = float(start_lat)
        start_lng = float(start_lng)

        nearest_index = min(
            range(len(unvisited)),
            key=lambda i: calculate_distance(
                start_lat,
                start_lng,
                float(unvisited[i]["y"]),
                float(unvisited[i]["x"])
            )
        )

        current = unvisited.pop(nearest_index)
    else:
        # 현재 위치가 없으면 첫 번째 장소부터 시작
        current = unvisited.pop(0)

    optimized_route.append(current)

    # 4. 현재 장소에서 가장 가까운 장소 찾기
    while unvisited:
        current_lat = float(current.get("y"))
        current_lng = float(current.get("x"))

        nearest_index = None
        min_distance = float("inf")

        for i, place in enumerate(unvisited):
            p_lat = float(place.get("y"))
            p_lng = float(place.get("x"))

            dist = calculate_distance(
                current_lat,
                current_lng,
                p_lat,
                p_lng
            )

            if dist < min_distance:
                min_distance = dist
                nearest_index = i

        # 다음 장소를 찾지 못한 경우 안전하게 종료
        if nearest_index is None:
            optimized_route.extend(unvisited)
            break

        # 5. 가장 가까운 장소를 경로에 추가
        next_place = unvisited.pop(nearest_index)
        next_place["distance_from_prev_km"] = round(min_distance, 2)

        optimized_route.append(next_place)
        current = next_place

    # 6. 좌표가 없는 장소는 마지막에 배치
    optimized_route.extend(places_without_coords)

    return optimized_route
