import pint

ureg = pint.UnitRegistry()
s = ureg.s          # секунда как единица
Hz = ureg.Hz        # герц

result = s * Hz * s
print(result)       # 1 (безразмерная единица)
print(result.dimensionality)  # 1 (безразмерно)