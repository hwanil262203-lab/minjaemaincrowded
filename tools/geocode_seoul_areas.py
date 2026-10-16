#!/usr/bin/env python3
"""서울 실시간 도시데이터 121개 장소 → 위경도 표 (seoul_areas.json) 생성.
Nominatim(OpenStreetMap) 으로 장소명을 지오코딩한다. 1초에 1회 제한.
실패한 장소는 lat/lon 이 null 로 남으며 손으로 채우면 된다.
"""
import json, sys, time, re, urllib.parse, urllib.request
sys.stdout.reconfigure(encoding="utf-8")

AREAS = """POI001 강남 MICE 관광특구
POI002 동대문 관광특구
POI003 명동 관광특구
POI004 이태원 관광특구
POI005 잠실 관광특구
POI006 종로·청계 관광특구
POI007 홍대 관광특구
POI008 경복궁
POI009 광화문·덕수궁
POI010 보신각
POI011 서울 암사동 유적
POI012 창덕궁·종묘
POI013 가산디지털단지역
POI014 강남역
POI015 건대입구역
POI016 고덕역
POI017 고속터미널역
POI018 교대역
POI019 구로디지털단지역
POI020 구로역
POI021 군자역
POI023 대림역
POI024 동대문역
POI025 뚝섬역
POI026 미아사거리역
POI027 발산역
POI029 사당역
POI030 삼각지역
POI031 서울대입구역
POI032 서울식물원·마곡나루역
POI033 서울역
POI034 선릉역
POI035 성신여대입구역
POI036 수유역
POI037 신논현역·논현역
POI038 신도림역
POI039 신림역
POI040 신촌·이대역
POI041 양재역
POI042 역삼역
POI043 연신내역
POI044 오목교역·목동운동장
POI045 왕십리역
POI046 용산역
POI047 이태원역
POI048 장지역
POI049 장한평역
POI050 천호역
POI051 총신대입구(이수)역
POI052 충정로역
POI053 합정역
POI054 혜화역
POI055 홍대입구역(2호선)
POI056 회기역
POI058 가락시장
POI059 가로수길
POI060 광장(전통)시장
POI061 김포공항
POI063 노량진
POI064 덕수궁길·정동길
POI066 북촌한옥마을
POI067 서촌
POI068 성수카페거리
POI070 쌍문역
POI071 압구정로데오거리
POI072 여의도
POI073 연남동
POI074 영등포 타임스퀘어
POI076 용리단길
POI077 이태원 앤틱가구거리
POI078 인사동
POI079 창동 신경제 중심지
POI080 청담동 명품거리
POI081 청량리 제기동 일대 전통시장
POI082 해방촌·경리단길
POI083 DDP(동대문디자인플라자)
POI084 DMC(디지털미디어시티)
POI085 강서한강공원
POI086 고척돔
POI087 광나루한강공원
POI088 광화문광장
POI089 국립중앙박물관·용산가족공원
POI090 난지한강공원
POI091 남산공원
POI092 노들섬
POI093 뚝섬한강공원
POI094 망원한강공원
POI095 반포한강공원
POI096 북서울꿈의숲
POI098 서리풀공원·몽마르뜨공원
POI100 서울대공원
POI101 서울숲공원
POI102 아차산
POI103 양화한강공원
POI104 어린이대공원
POI105 여의도한강공원
POI106 월드컵공원
POI107 응봉산
POI108 이촌한강공원
POI109 잠실종합운동장
POI110 잠실한강공원
POI111 잠원한강공원
POI112 청계산
POI114 북창동 먹자골목
POI115 남대문시장
POI116 익선동
POI117 신정네거리역
POI118 잠실새내역
POI119 잠실역
POI120 잠실롯데타워·석촌호수
POI121 송리단길·호수단길
POI122 신촌 스타광장
POI123 보라매공원
POI124 서대문독립공원
POI125 안양천
POI126 여의서로
POI127 올림픽공원
POI128 홍제폭포
POI129 송현녹지광장
POI130 시의회 앞
POI131 숭례문"""

# 지오코딩이 어려운 이름의 검색어 보정
ALIAS = {
    "강남 MICE 관광특구": "코엑스", "동대문 관광특구": "동대문디자인플라자", "명동 관광특구": "명동역", "이태원 관광특구": "이태원역",
    "잠실 관광특구": "잠실역", "종로·청계 관광특구": "종각역", "홍대 관광특구": "홍대입구역", "서울 암사동 유적": "암사동 선사유적지",
    "DDP(동대문디자인플라자)": "동대문디자인플라자", "DMC(디지털미디어시티)": "디지털미디어시티역", "총신대입구(이수)역": "이수역",
    "홍대입구역(2호선)": "홍대입구역", "광장(전통)시장": "광장시장", "창동 신경제 중심지": "창동역", "청량리 제기동 일대 전통시장": "청량리역",
    "북창동 먹자골목": "북창동", "시의회 앞": "서울특별시의회", "여의서로": "여의서로", "용리단길": "신용산역", "송리단길·호수단길": "석촌호수",
    "잠실롯데타워·석촌호수": "롯데월드타워", "신촌 스타광장": "신촌역", "덕수궁길·정동길": "정동길", "해방촌·경리단길": "경리단길",
    "서리풀공원·몽마르뜨공원": "서리풀공원", "국립중앙박물관·용산가족공원": "국립중앙박물관", "오목교역·목동운동장": "오목교역",
    "신논현역·논현역": "신논현역", "신촌·이대역": "신촌역", "서울식물원·마곡나루역": "마곡나루역", "광화문·덕수궁": "덕수궁", "창덕궁·종묘": "창덕궁",
    "청담동 명품거리": "청담역", "이태원 앤틱가구거리": "이태원 앤틱가구거리", "영등포 타임스퀘어": "타임스퀘어 영등포", "압구정로데오거리": "압구정로데오역",
}

def geocode(q):
    url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode({"format": "json", "limit": 1, "countrycodes": "kr", "accept-language": "ko", "q": q})
    req = urllib.request.Request(url, headers={"User-Agent": "minjae-gps-prototype/0.1 (school project)"})
    with urllib.request.urlopen(req, timeout=20) as r:
        js = json.load(r)
    return (float(js[0]["lat"]), float(js[0]["lon"]), js[0]["display_name"]) if js else None

out = []
for line in AREAS.splitlines():
    code, name = line.split(" ", 1)
    tries = [ALIAS.get(name, name) + " 서울", ALIAS.get(name, name), re.split(r"[·(]", name)[0] + " 서울"]
    hit = None
    for q in tries:
        try:
            hit = geocode(q)
        except Exception as e:
            hit = None
        time.sleep(1.1)
        if hit:
            break
    out.append({"code": code, "name": name, "lat": hit[0] if hit else None, "lon": hit[1] if hit else None, "query": q if hit else None, "matched": hit[2] if hit else None})
    print(f"{code} {name:<24} -> {'OK' if hit else 'FAIL'} {hit[2][:50] if hit else ''}")

with open("seoul_areas.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("saved seoul_areas.json  OK:", sum(1 for o in out if o["lat"]), "/", len(out))
