from W.PySide6.Data.Reactive import ReactiveDict

if __name__ == "__main__":
    d = ReactiveDict({"a": 1})
    d.signals.itemSet.connect(lambda k, nv, ov: print(f"Changed {k}: {ov} -> {nv}"))
    d["a"] = 2
    d["b"] = 3
