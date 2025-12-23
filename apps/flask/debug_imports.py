"""Debug script untuk menemukan import error"""

print("=" * 60)
print("Testing imports...")
print("=" * 60)

try:
    print("\n1. Testing helpers.objectid_converter...")
    from helpers.objectid_converter import convert_objectid
    print("   ✅ OK")
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

try:
    print("\n2. Testing jobs.run_forecast_from_config...")
    from jobs.run_forecast_from_config import run_forecast_from_config
    print("   ✅ OK")
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

try:
    print("\n3. Testing jobs.run_lstm...")
    from jobs.run_lstm import run_lstm_from_config
    print("   ✅ OK")
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

try:
    print("\n4. Testing holt_winter.hw_dynamic...")
    from holt_winter.hw_dynamic import run_optimized_hw_analysis
    print("   ✅ OK")
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

try:
    print("\n5. Testing lstm.lstm_dynamic_2...")
    from lstm.lstm_dynamic_2 import run_lstm_analysis
    print("   ✅ OK")
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Debug complete!")
print("=" * 60)