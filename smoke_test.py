from trinity import Engine, default_config

cfg = default_config()
cfg["tiers"][2]["enabled"] = True
engine = Engine(cfg)
solution = engine.solve()
stats = engine.statistics(570.0, 0.0)

assert solution["converged"]
assert abs(solution["averageReward"] - 5.01762e-05) < 2e-10
assert abs(stats["meanBase"] - 4603.94) < 2.0
assert abs(stats["stdBase"] - 12451.4) < 2.0

print(f"Howard iterations: {solution['iterations']}")
print(f"Average reward: {solution['averageReward']:.8g}")
print(f"Closed-form mean: {stats['meanBase']:.2f} EUR")
print(f"Closed-form stdev: {stats['stdBase']:.2f} EUR")
print("Python/C++ smoke test passed.")
