import sys
sys.path.append('/run/media/cryptedlm/local_d/Kuliah/Tugas Akhir/Repository/tumbuh-baik/apps/flask')

from services.kecamatan_kabupatan_mapping_service import KecamatanKabupatenMappingService

def test_mapping_service():
    """Test the Kecamatan-Kabupaten Mapping Service"""
    print("🧪 Testing Kecamatan-Kabupaten Mapping Service...")
    
    try:
        # Initialize service
        mapping_service = KecamatanKabupatenMappingService()
        print("✅ Mapping service initialized")
        
        # Test kecamatan info
        print("\n📍 Testing kecamatan info...")
        test_kecamatan = ["Lhoksukon", "Indrapuri", "Juli", "Pidie"]
        
        for kec in test_kecamatan:
            info = mapping_service.get_kecamatan_info(kec)
            if info:
                print(f"   {kec} → {info.kecamatan_name} ({info.kabupaten_name})")
                print(f"      Area: {info.area_km2} km² (weight: {info.area_weight})")
                print(f"      Codes: Kec={info.kecamatan_code}, Kab={info.kabupaten_code}")
                if info.nasa_location_name:
                    print(f"      NASA location: {info.nasa_location_name}")
        
        # Test kabupaten info
        print("\n🏛️ Testing kabupaten info...")
        for kabupaten in mapping_service.get_all_kabupaten():
            info = mapping_service.get_kabupaten_info(kabupaten)
            if info:
                print(f"   {kabupaten} → BPS: {info.bps_compatible_name}")
                print(f"      Area: {info.total_area_km2} km², {info.kecamatan_count} kecamatan")
                print(f"      Kecamatan: {info.constituent_kecamatan}")
        
        # Test validation
        print("\n✅ Running validation...")
        validation = mapping_service.validate_mapping_consistency()
        print(f"   Kabupaten: {validation['total_kabupaten']}")
        print(f"   Kecamatan: {validation['total_kecamatan']}")
        print(f"   NASA locations: {validation['total_nasa_locations']}")
        
        # Check BPS mapping
        print("\n🔗 BPS mapping validation:")
        for internal, bps_name in mapping_service.get_bps_compatible_kabupaten_names().items():
            print(f"   {internal} → {bps_name}")
        
        print("\n🎉 Mapping service test completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_mapping_service()