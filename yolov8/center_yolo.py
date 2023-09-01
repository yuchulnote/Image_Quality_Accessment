import matplotlib.pyplot as plt


def compute_area(coords):
    """주어진 좌표를 기반으로 다각형의 면적을 계산합니다."""
    n = len(coords) // 2
    area = 0.0
    for i in range(n):
        x1, y1 = coords[i * 2], coords[i * 2 + 1]
        x2, y2 = coords[(i * 2 + 2) % (n * 2)], coords[(i * 2 + 3) % (n * 2)]
        area += (x1 * y2) - (x2 * y1)
    return abs(area) / 2.0


def extract_center_coords(coords):
    xs = coords[::2]
    ys = coords[1::2]

    center_coords = []
    for x, y in zip(xs, ys):
        if 0.4 <= x <= 0.6:
            center_coords.extend([x, y])

    return center_coords


def classify_polygon(data):
    classes, coords = [], []

    for line in data:
        parts = line.split()
        cls = int(parts[0])
        coordinates = list(map(float, parts[1:]))

        classes.append(cls)
        coords.append(coordinates)

    if len(classes) < 3:
        print(f"인스턴스 수: {classes}, 3개 안 되어서 거절")
        return False

    poly_0 = [coord for i, coord in enumerate(coords) if classes[i] == 0]
    poly_1 = [coord for i, coord in enumerate(coords) if classes[i] == 1]

    if len(poly_0) != 2 or not poly_1:
        print(f"콧구멍 수: {poly_0}, 인중 수: {poly_1} 거절. (2, 1) 이상")
        return False

    area_0_1 = compute_area(poly_0[0])
    area_0_2 = compute_area(poly_0[1])
    area_diff_criteria = 1.2
    area_diff = area_0_1 / area_0_2
    area_diff = 1 / area_diff if area_diff < 1 else area_diff

    if area_diff > area_diff_criteria:
        print(f"콧구멍 차이: {area_diff}. {area_diff_criteria} 이상이라 거절")

        return False

    mid_area = compute_area(poly_1[0])
    philtrum_coords = extract_center_coords(poly_1[0])
    philtrum_area = compute_area(philtrum_coords)
    mid_area_ratio = philtrum_area / mid_area

    if mid_area_ratio < 0.9:
        return False

    # print(
    #    f"인중mask 넓이:{mid_area}, 중앙 해당 인중 넓이: {philtrum_area}, 중앙 헤당 인중 비율: {mid_area_ratio}"
    # )
    return True


def plot_polygon(data):
    all_ys = []
    for line in data:
        parts = line.split()
        coordinates = list(map(float, parts[1:]))
        ys = coordinates[1::2]
        all_ys.extend(ys)

    max_y = max(all_ys)

    for line in data:
        parts = line.split()
        cls = int(parts[0])
        coordinates = list(map(float, parts[1:]))

        xs = coordinates[::2]
        ys = [max_y - y for y in coordinates[1::2]]  # Y값을 반전시키는 부분

        if cls == 0:
            plt.fill(xs, ys, "b-", alpha=0.6)  # blue for class 0
        else:
            plt.fill(xs, ys, "r-", alpha=0.6)  # red for class 1

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Polygon Visualization")
    plt.grid(True)
    plt.show()


# File path
predict_path = (
    # 욜로 출력 txt 디렉토리
)

# Read data from the file
with open(predict_path, "r") as f:
    data = f.readlines()

print(classify_polygon(data))
plot_polygon(data)


# 욜로 출력시 arg 참고
# save_txt=True show_labels=False show_conf=False boxes=False save=True iou=0
