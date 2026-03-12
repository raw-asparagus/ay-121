def table(result, name: str):
    return result.tables[name]


def table_names(result) -> list[str]:
    return sorted(result.tables)
