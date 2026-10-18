import math

def calculate_distance(lat1, lon1, lat2, lon2):
    """
    두 위도 / 경도 좌표 간의 거리 (km)를 계산 (Haversine 공식)
    """
    R = 6371.0
    lat1_rad, lon1_rad = math.radians(float(lat1)), math.radians(float(lon1))
    
    # 💡 아래 부분을 math.radians로 수정!
    lat2_rad, lon2_rad = math.radians(float(lat2)), math.radians(float(lon2))

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c

def optimize_route(places):
    """
    위치 (x, y) 기반 최단 동선 경로 (Nearest Neighbor)
    """
    if not places:
        return []
        
    unvisited = places.copy()
    optimized_route = []

    # 첫 번째 장소를 출발점으로 설정
    current = unvisited.pop(0)
    optimized_route.append(current)

    while unvisited:
        current_lat, current_lng = current.get("y"), current.get("x")
        
        # 좌표 정보가 없는 경우 안전하게 처리
        if not current_lat or not current_lng:
            optimized_route.extend(unvisited)
            break

        nearest_index = 0
        min_distance = float('inf')

        for i, place in enumerate(unvisited):
            p_lat, p_lng = place.get("y"), place.get("x")
            if not p_lat or not p_lng:
                continue

            dist = calculate_distance(current_lat, current_lng, p_lat, p_lng)
            if dist < min_distance:
                min_distance = dist
                nearest_index = i

        next_place = unvisited.pop(nearest_index)
        next_place["distance_from_prev_km"] = round(min_distance, 2)
        optimized_route.append(next_place)
        current = next_place
        
    return optimized_route