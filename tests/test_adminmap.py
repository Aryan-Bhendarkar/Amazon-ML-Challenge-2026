from ber import adminmap
from ber.normalize import normalize_address


def _data():
    cities = {"alpha": "north region", "beta": "north region", "gamma": "north region", "delta": "south region",
              "eps": "south region"}
    dept = {"alpha": "upper dept", "beta": "upper dept", "gamma": "lower dept", "delta": "far dept", "eps": "far dept"}
    s1, other = [], []
    for i in range(3000):
        c = list(cities)[i % 5]
        s1.append(f"{i} Main St, {c.title()}, {cities[c].title()}")
        other.append(f"{i} MAIN ST, {c.upper()}, {dept[c].upper()}" if i % 2 else f"{i} Main St, {c}")
    return s1, other


def test_learns_department_aliases_from_records_only():
    m = adminmap.learn(*_data(), min_count=50)
    assert m["admins"] == ["north region", "south region"]
    assert m["alias"] == {"far dept": "south region", "lower dept": "north region", "upper dept": "north region"}
    adm = {"admins": set(m["admins"]), "alias": m["alias"]}
    a = normalize_address("7 Main St, Alpha, North Region", "Xland", adm)
    b = normalize_address("7 MAIN ST, ALPHA, UPPER DEPT", "Xland", adm)
    assert a.state == b.state == "north region" and "dept" not in b.street and "region" not in a.street


def test_parser_countries_unchanged_without_map():
    assert normalize_address("12 Nord Ave, Chicago, Illinois", "US").state == "il"
