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


def test_france_v2_address_state_and_bis():
    # region (every test S1) and department (31% of pool) map to the same state; bis/ter dropped
    a = normalize_address("4 Rue Roger Salengro, Saint-Nazaire, Pays de la Loire", "France")
    b = normalize_address("4 R Roger Salengro, St-Nazaire, Loire-Atlantique", "France")
    assert a.state == b.state == "pdl" and a.full == b.full
    assert "loire" not in a.street and "pays" not in a.street
    assert normalize_address("1BIS RUE CHARLES DELESALLE, Lille, Nord", "France").house == "1"
    assert normalize_address("15 ter Rue X, Lille, Hauts-de-France", "France").state == "hdf"


def test_france_v2_name_rules():
    n = normalize_name("Chasse Collège (France) S.A.S", country="France")
    assert n.core == "chasse college" and n.legal == "sas"
    assert normalize_name("Martin et Frs Cie", country="France").core == \
        normalize_name("Martin & Freres Compagnie", country="France").core == "martin freres"
    assert normalize_name("Dupont EI", country="France").legal == "ei"


def test_v2_rules_are_country_keyed():
    # generic fallback: unknown / other countries unchanged (US/India byte-identical to norm v1)
    assert normalize_name("Martin et Cie (France)", country="US") == normalize_name("Martin et Cie (France)")
    assert normalize_name("Martin et Cie", country="Germany").core == "martin et cie"
    assert normalize_address("12 Main St, Paris, Nord", "US").state == ""
    assert normalize_address("1 bis Rue X, Lille, Nord", "").house == "1"   # no country: no FR rules
    assert normalize_address("1bis Rue X, Lille, Nord", "").state == ""
