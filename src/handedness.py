"""Player handedness for the 109 servers in the dataset."""

LEFT_HANDED = {
    "Denis Shapovalov",
    "Beatriz Haddad Maia",
}


def hand(server: str) -> str:
    return "L" if server in LEFT_HANDED else "R"
