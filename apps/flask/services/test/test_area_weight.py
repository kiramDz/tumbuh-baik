import sys
import os

# Add the flask app directory to Python path
sys.path.append('/run/media/cryptedlm/local_d/Kuliah/Tugas Akhir/Repository/tumbuh-baik/apps/flask')

from services.area_weight_service import AreaWeightService

def test_area_weight_service():
    """Test the Area Weight Service"""
    print("🧪 Testing Area Weight Service...")
    
    try:
        # Initialize service
        print("\n1. Initializing Area Weight Service...")
        area_service = AreaWeightService()
        print("✅ Service initialized successfully")
        
        # Test basic information
        print("\n2. Testing basic information...")
        all_kabupaten = area_service.get_all_kabupaten()
        print(f"📊 Found {len(all_kabupaten)} kabupaten: {all_kabupaten}")
        
        total_kecamatan = len(area_service.kecamatan_info)
        print(f"📊 Found {total_kecamatan} kecamatan total")
        
        # Test area weights for each kabupaten
        print("\n3. Testing area weights by kabupaten...")
        for kabupaten in all_kabupaten:
            weights = area_service.get_area_weights_for_kabupaten(kabupaten)
            total_area = area_service.get_kabupaten_total_area(kabupaten)
            weight_sum = sum(weights.values())
            
            print(f"\n📍 {kabupaten}:")
            print(f"   Total area: {total_area} km²")
            print(f"   Kecamatan count: {len(weights)}")
            print(f"   Weight sum: {weight_sum:.4f} (should be ~1.0)")
            print(f"   Kecamatan weights:")
            
            for kecamatan, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
                kecamatan_info = area_service.get_kecamatan_area_info(kecamatan)
                area_km2 = kecamatan_info.area_km2 if kecamatan_info else 0
                percentage = weight * 100
                print(f"     • {kecamatan}: {weight:.4f} ({percentage:.1f}%) - {area_km2} km²")
        
        # Test NASA location mapping
        print("\n4. Testing NASA location mapping...")
        nasa_mapping = area_service.get_nasa_location_mapping()
        print(f"📡 NASA locations mapped: {len(nasa_mapping)}")
        
        for nasa_location, info in nasa_mapping.items():
            print(f"   {nasa_location} → {info['kecamatan_name']} ({info['kabupaten_name']}) - {info['area_weight']:.4f}")
        
        # Test validation
        print("\n5. Running validation...")
        validation = area_service.validate_area_weights()
        print(f"📋 Validation results:")
        print(f"   Total kabupaten: {validation['total_kabupaten']}")
        print(f"   Total kecamatan: {validation['total_kecamatan']}")
        
        # Check weight sum validation
        print("\n   Weight sum validation:")
        for kabupaten, validation_info in validation['weight_sum_validation'].items():
            status = "✅ VALID" if validation_info['valid'] else "❌ INVALID"
            print(f"   {kabupaten}: {validation_info['actual']} {status}")
        
        # Test summary
        print("\n6. Getting area weight summary...")
        summary = area_service.get_area_weight_summary()
        print(f"📈 Summary generated for {summary['total_kabupaten']} kabupaten")
        
        print("\n🎉 All tests completed successfully!")
        return True
        
    except FileNotFoundError as e:
        print(f"❌ GeoJSON file not found: {e}")
        print("💡 Make sure the GeoJSON file exists at the specified path")
        return False
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_area_weight_service()