from ber.normalize import normalize_address, normalize_name


def test_name_noise_removed():
    assert normalize_name("regional prime horizon center (ID: 96415)").core == "regional prime horizon center"
    assert normalize_name("Regional Prime Horizon-Center #70318").core == "regional prime horizon center"
    assert normalize_name("-- Energy Vr Textile Corporation").legal == "corp"
    assert normalize_name("G1obal Dte L.L.C. L.L.C.").legal == "llc"
    assert normalize_name("NA").kind == "empty"


def test_domains_handles_alias():
    assert normalize_name("energyvrtextile.com").compact == "energyvrtextile"
    assert normalize_name("@SORCHASPIZZA").kind == "handle"
    assert normalize_name("Deltatavo formerly known as Niex Holding Company Center").alias == "niex holding center"


def test_transliteration_and_legal():
    n = normalize_name("ब्लू इम्पेक्स प्राइवेट लिमिटेड")
    assert n.legal == "ltd pvt" and n.core.startswith("blu")


def test_address():
    a = normalize_address("##801 Broad Street, PMB 9914, Winston-Salem, North Carolina", "US")
    assert (a.house, a.state) == ("801", "nc") and "st" in a.full.split()
    f = normalize_address("N° 50 R. DE LA BENAUGE, BORDEAUX, Nouvelle-Aquitaine", "France")
    assert f.house == "50" and "rue" in f.full.split()
    assert normalize_address("Plot 12, Sector 5, Noida, UP 201301", "India").postcode == "201301"
    assert normalize_address("", "US").empty
    # unknown country must not crash
    assert normalize_address("12 Some St, Town", "Germany").house == "12"
