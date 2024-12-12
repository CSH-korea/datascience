import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pycountry_convert as pc
import plotly.express as px
import numpy as np

# CSV 파일 로드
file_path = "C:\\Users\\82104\\Desktop\\football-data-analysis-main\\fifa_players.csv"  # CSV 파일 경로
df = pd.read_csv(file_path)

# 데이터 확인
print(df.head())
print(df.info())

print(df.isnull().sum())  # 각 열의 결측값 개수 확인

df.fillna(0, inplace=True)  # 결측값을 0으로 채움

#수치형으로 변환이 필요한 열을 변환
df['age'] = pd.to_numeric(df['age'], errors='coerce')
df['height_cm'] = pd.to_numeric(df['height_cm'], errors='coerce')

# 1.선수 능력치와 가치 간의 상관관계 분석
correlation = df[['overall_rating', 'potential', 'value_euro', 'wage_euro']].corr()
print(correlation)
# 1 시각화
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title("Correlation Between Attributes and Value")
plt.show()


# 2. 포지션별 평균계산 - 포지션별로 그룹화하여 평균 계산
# 포지션 그룹 설정 함수는 기존 코드 사용
def map_position(position):
    if position in ['ST', 'CF', 'LW', 'RW']:  # 공격수 포지션
        return 'Forwards'
    elif position in ['CM', 'CAM', 'CDM', 'LM', 'RM']:  # 미드필더 포지션
        return 'Midfielders'
    elif position in ['CB', 'LB', 'RB', 'LWB', 'RWB']:  # 수비수 포지션
        return 'Defenders'
    elif position == 'GK':  # 골키퍼 포지션
        return 'Goalkeepers'
    else:
        return 'Others'  # 기타

# 기존 포지션 그룹 생성 코드
# 새로운 열 생성
df['position_group'] = df['positions'].apply(map_position)

# 포지션 그룹별 분포 확인
print(df['position_group'].value_counts())

# 포지션 그룹별 평균 계산
group_stats = df.groupby('position_group')[['overall_rating', 'potential', 'value_euro', 'wage_euro']].mean()

# 값 확인
print(group_stats)
filtered_stats = group_stats.drop(index='Others', errors='ignore')

# 기존 시각화 코드: 포지션 그룹별 평균 시장 가치
filtered_stats['value_euro'].sort_values(ascending=False).plot(kind='bar', figsize=(8, 6))
plt.title("Average Market Value by Position Group")
plt.ylabel("Market Value (Euro)")
plt.xlabel("Position Group")
plt.xticks(rotation=0)
plt.show()

# 기존 시각화 코드: 포지션 그룹별 평균 주급
filtered_stats['wage_euro'].sort_values(ascending=False).plot(kind='bar', figsize=(8, 6))
plt.title("Average Wage by Position Group")
plt.ylabel("Average Wage (Euro)")
plt.xlabel("Position Group")
plt.xticks(rotation=0)
plt.show()

# 추가 분석 및 시각화: 포지션별 주요 능력치 비교
# 주요 능력치 정의
position_important_attributes = {
    "Forwards": ["finishing", "sprint_speed", "dribbling"],
    "Midfielders": ["short_passing", "vision", "ball_control"],
    "Defenders": ["standing_tackle", "sliding_tackle", "strength"],
    "Goalkeepers": ["gk_diving", "gk_handling", "gk_positioning"]
}

# 포지션별 주요 능력치 평균 계산
attribute_comparison = {}
for position, attributes in position_important_attributes.items():
    if all(attr in df.columns for attr in attributes):  # 필요한 열이 모두 존재하는 경우에만 실행
        attribute_comparison[position] = df[df['position_group'] == position][attributes].mean()

# 데이터프레임으로 변환
attribute_comparison_df = pd.DataFrame(attribute_comparison).T  # 전치하여 보기 쉽게 정리
print(attribute_comparison_df)

# 포지션별 주요 능력치 비교 시각화
attribute_comparison_df.plot(kind='bar', figsize=(12, 8), colormap='viridis', edgecolor='black')
plt.title("Position-wise Key Attribute Comparison")
plt.ylabel("Average Attribute Value")
plt.xlabel("Position Group")
plt.xticks(rotation=0)
plt.legend(title="Attributes", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 추가 분석 및 시각화: 포지션별 Overall Rating 비교
# 포지션별 Overall Rating 평균 계산
position_overall_rating = df.groupby('position_group')['overall_rating'].mean()

# 시각화: 포지션별 Overall Rating
position_overall_rating.sort_values(ascending=False).plot(kind='bar', figsize=(8, 6), color='skyblue', edgecolor='black')
plt.title("Average Overall Rating by Position Group")
plt.ylabel("Average Overall Rating")
plt.xlabel("Position Group")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()



# 3. 대륙별 선수 데이터 분석
# 3. 대륙별 선수 데이터 분석
import pycountry_convert as pc
import pandas as pd
import plotly.express as px

# 국가를 대륙으로 변환하는 함수
def country_to_continent(country_name):
    try:
        # 국가 코드 가져오기
        country_alpha2 = pc.country_name_to_country_alpha2(country_name)
        # 대륙 코드 가져오기
        continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
        # 대륙 코드로 대륙 이름 반환
        continents = {
            'AF': 'Africa',
            'AS': 'Asia',
            'EU': 'Europe',
            'NA': 'North America',
            'SA': 'South America',
            'OC': 'Oceania',
            'AN': 'Antarctica'
        }
        return continents[continent_code]
    except:
        # 국가 매핑 실패 시
        return 'Unknown'

# 대륙 열 추가
df['continent'] = df['nationality'].apply(country_to_continent)

# 대륙별 데이터 확인
print("대륙별 선수 수:")
print(df['continent'].value_counts())

# Unknown 제외
filtered_df = df[df['continent'] != 'Unknown']

# 대륙별 평균 계산
continent_stats = filtered_df.groupby('continent')[['overall_rating', 'value_euro', 'wage_euro']].mean().reset_index()
print("대륙별 평균 데이터:")
print(continent_stats)

# 대륙별 중심 좌표 설정 (위도, 경도)
continent_geo_data = {
    'Africa': {'lat': 0, 'lon': 20},
    'Asia': {'lat': 40, 'lon': 100},
    'Europe': {'lat': 54, 'lon': 15},
    'North America': {'lat': 40, 'lon': -100},
    'South America': {'lat': -15, 'lon': -60},
    'Oceania': {'lat': -25, 'lon': 140}
}

# 좌표 정보를 continent_stats에 추가
continent_stats['lat'] = continent_stats['continent'].map(lambda x: continent_geo_data.get(x, {}).get('lat', None))
continent_stats['lon'] = continent_stats['continent'].map(lambda x: continent_geo_data.get(x, {}).get('lon', None))

# 좌표가 없는 경우 확인
if continent_stats[['lat', 'lon']].isnull().any().any():
    print("경고: 일부 대륙에 좌표가 설정되지 않았습니다.")
    print(continent_stats[continent_stats[['lat', 'lon']].isnull().any(axis=1)])

# 지도 시각화 (능력치)
fig = px.scatter_geo(
    continent_stats,
    lat='lat',
    lon='lon',
    text='continent',  # 대륙 이름 표시
    size='overall_rating',  # 능력치 크기로 표현
    size_max=80,  # 원의 최대 크기 설정
    color='overall_rating',  # 색상으로 능력치 표현
    title="Average Overall Rating by Continent",
    projection='natural earth',  # 지구 투영 방식
    labels={'overall_rating': 'Average Overall Rating'},
    color_continuous_scale=px.colors.sequential.Reds  # 빨간색을 강조한 색상 스케일
)

fig.update_geos(showcountries=True, showcoastlines=True, showland=True, landcolor="LightGreen")
fig.show()

# 지도 시각화 (시장 가치)
fig = px.scatter_geo(
    continent_stats,
    lat='lat',
    lon='lon',
    text='continent',  # 대륙 이름 표시
    size='value_euro',  # 시장 가치 크기로 표현
    size_max=80,  # 원의 최대 크기 설정
    color='value_euro',  # 색상으로 시장 가치 표현
    title="Average Market Value by Continent",
    projection='natural earth',  # 지구 투영 방식
    labels={'value_euro': 'Average Market Value (Euro)'},
    color_continuous_scale=px.colors.sequential.Reds  # 빨간색을 강조한 색상 스케일
)

fig.update_geos(showcountries=True, showcoastlines=True, showland=True, landcolor="LightGreen")
fig.show()

# 지도 시각화 (주급)
fig = px.scatter_geo(
    continent_stats,
    lat='lat',
    lon='lon',
    text='continent',  # 대륙 이름 표시
    size='wage_euro',  # 주급 크기로 표현
    size_max=80,  # 원의 최대 크기 설정
    color='wage_euro',  # 색상으로 주급 표현
    title="Average Wage by Continent",
    projection='natural earth',  # 지구 투영 방식
    labels={'wage_euro': 'Average Wage (Euro)'},
    color_continuous_scale=px.colors.sequential.Reds  # 빨간색을 강조한 색상 스케일
)

fig.update_geos(showcountries=True, showcoastlines=True, showland=True, landcolor="LightGreen")
fig.show()


#4. international reputation과 overall rating간의 관계 
import matplotlib.pyplot as plt
import numpy as np

# 데이터 준비
reputation_rating = df[['international_reputation(1-5)', 'overall_rating']].copy()
reputation_rating.columns = ['international_reputation', 'overall_rating']

# 국제 명성별 평균 overall_rating 계산
grouped_data = reputation_rating.groupby('international_reputation')['overall_rating'].mean().reset_index()

# 방사형 차트 데이터 준비
angles = np.linspace(0, 2 * np.pi, len(grouped_data), endpoint=False).tolist()
angles += angles[:1]  # 마지막 값을 닫아주는 처리

# 데이터 연결
values = grouped_data['overall_rating'].tolist()
values += values[:1]

# 방사형 차트 생성
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# 데이터 그리기
ax.fill(angles, values, color='skyblue', alpha=0.5, edgecolor='black')  # 영역 채우기
ax.plot(angles, values, color='blue', linewidth=2)  # 데이터 연결 선

# 값 표시
for i, (angle, value) in enumerate(zip(angles, values)):
    if i < len(values) - 1:  # 마지막 값은 표시하지 않음(중복)
        ax.text(
            angle, value + 1, f'{value:.1f}',  # 값 표시
            ha='center', va='center', fontsize=10, color='darkblue', fontweight='bold'
        )

# 축 설정
ax.set_yticks(range(60, int(max(values)) + 5, 1))  # y축 범위를 좁게 설정
ax.set_yticklabels([])  # y축 레이블 제거
ax.set_xticks(angles[:-1])  # x축 각도 설정
ax.set_xticklabels(grouped_data['international_reputation'].astype(str))  # x축 레이블

# 제목 설정
ax.set_title("Relationship Between International Reputation and Overall Rating", va='bottom', fontsize=14)

plt.show()










#5 - 그렇다면 왜 유저들은 게임의 능력치가 현실반영이 잘 되지 않았다고 느낄까?
#원인분석 1
# 주요 포지션별 중요 능력치 정의 (골키퍼 열 제거)
position_important_attributes = {
    "Forwards": ["finishing", "sprint_speed", "dribbling"],
    "Midfielders": ["short_passing", "ball_control", "vision"],
    "Defenders": ["standing_tackle", "sliding_tackle", "strength"],
}

# 포지션 그룹화 함수
def map_position(position):
    if position in ['ST', 'CF', 'LW', 'RW']:
        return 'Forwards'
    elif position in ['CM', 'CAM', 'CDM', 'LM', 'RM']:
        return 'Midfielders'
    elif position in ['CB', 'LB', 'RB', 'LWB', 'RWB']:
        return 'Defenders'
    elif position == 'GK':
        return 'Goalkeepers'
    else:
        return 'Others'

# 포지션 그룹 생성
df['position_group'] = df['positions'].apply(map_position)

# 포지션별로 데이터 필터링
correlations = {}
for position, attributes in position_important_attributes.items():
    # 존재하는 열만 선택
    valid_attributes = [attr for attr in attributes if attr in df.columns]
    subset = df[df['position_group'] == position][valid_attributes + ['overall_rating']]
    correlations[position] = subset.corr()

# 상관관계 확인 (Forwards 예시 출력)
print("Correlation for Forwards:")
print(correlations["Forwards"])

for position, corr_matrix in correlations.items():
    plt.figure(figsize=(10, 6))
    
    # overall_rating과 다른 열 간의 상관관계 추출
    corr_with_overall = corr_matrix['overall_rating'].drop('overall_rating')
    attributes = corr_with_overall.index.tolist()
    correlations = corr_with_overall.values

    # 버블 차트 생성
    plt.scatter(attributes, correlations, s=np.abs(correlations) * 1000, c=correlations, 
                cmap='coolwarm', edgecolor='black', alpha=0.7)
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)  # 0 기준선
    plt.colorbar(label='Correlation Coefficient')
    plt.title(f"Bubble Chart of Correlation with Overall Rating for {position}")
    plt.xlabel("Attributes")
    plt.ylabel("Correlation Coefficient")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


#원인분석 2
# 포지션별 주요 능력치 정의
position_important_attributes = {
    "Forwards": ["finishing", "sprint_speed", "dribbling"],
    "Midfielders": ["short_passing", "vision", "ball_control"],
    "Defenders": ["standing_tackle", "sliding_tackle", "strength"],
    "Goalkeepers": ["gk_diving", "gk_handling", "gk_positioning"]
}

# 포지션 그룹화 함수
def map_position(position):
    if position in ['ST', 'CF', 'LW', 'RW']:
        return 'Forwards'
    elif position in ['CM', 'CAM', 'CDM', 'LM', 'RM']:
        return 'Midfielders'
    elif position in ['CB', 'LB', 'RB', 'LWB', 'RWB']:
        return 'Defenders'
    elif position == 'GK':
        return 'Goalkeepers'
    else:
        return 'Others'

# 포지션 그룹 생성
df['position_group'] = df['positions'].apply(map_position)

# 4,500,000 유로 이상의 가치가 있는 선수들로 필터링
valuable_players = df[df['value_euro'] >= 4500000]

# 불균형 점수 계산 함수 (업데이트)
def calculate_disparity(df, position_group, attributes):
    subset = df[df['position_group'] == position_group]
    subset['attributes_avg'] = subset[attributes].mean(axis=1)
    subset['rating_disparity'] = subset['overall_rating'] - subset['attributes_avg']
    return subset.nlargest(10, 'rating_disparity')  # 불균형 점수가 높은 상위 10명 반환

# 각 포지션별 불균형 점수 계산 (필터링된 데이터 사용)
top_disparity_players = {}
for position, attributes in position_important_attributes.items():
    if all(attr in valuable_players.columns for attr in attributes):  # 필요한 열이 모두 존재하는 경우에만 실행
        top_disparity_players[position] = calculate_disparity(valuable_players, position, attributes)

# 결과 확인
for position, players in top_disparity_players.items():
    print(f"Top 10 Rating Disparity Players for {position}:")
    print(players[['name', 'overall_rating', 'attributes_avg', 'rating_disparity', 'value_euro']])

# 포지션별 막대 그래프 생성 (업데이트)
for position, players in top_disparity_players.items():
    plt.figure(figsize=(12, 6))
    plt.barh(players['name'], players['rating_disparity'], color='salmon', edgecolor='black')
    plt.xlabel("Rating Disparity (Overall Rating - Average Attributes)")
    plt.ylabel("Player Name")
    plt.title(f"Top 10 Rating Disparity Players for {position} (Value ≥ 4,500,000€)")
    plt.gca().invert_yaxis()  # 이름이 위에서 아래로 나열되도록 설정
    plt.tight_layout()
    plt.show()

# 버블 차트 생성 (업데이트)
for position, players in top_disparity_players.items():
    plt.figure(figsize=(10, 6))
    plt.scatter(players['attributes_avg'], players['overall_rating'],
                s=players['rating_disparity'] * 10, c=players['rating_disparity'],
                cmap='coolwarm', edgecolor='black', alpha=0.7)
    plt.colorbar(label="Rating Disparity")
    plt.xlabel("Average Key Attributes")
    plt.ylabel("Overall Rating")
    plt.title(f"Overall Rating vs Average Key Attributes for {position} (Value ≥ 4,500,000€)")
    plt.axline((0, 0), slope=1, color='gray', linestyle='--', linewidth=1, label="Equal Rating")
    plt.legend()
    plt.tight_layout()
    plt.show()
