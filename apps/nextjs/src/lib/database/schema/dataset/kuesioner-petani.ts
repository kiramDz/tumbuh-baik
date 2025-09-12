import { id } from "date-fns/locale";
import mongoose, { Schema } from "mongoose";

const KuesionerPetaniSchema = new Schema(
    {
        id_petani: { type: String, required: true },
        provinsi : { type: String, required: true },
        kab_kota : { type: String, required: true },
        kecamatan : { type: String, required: true },
        desa_gampong : { type: String, required: true },
        nama : { type: String, required: true },
        jenis_kelamin : { type: String, required: true },
        umur : { type: Number, required: true },
        alamat : { type: String, required: true },
        pendidikan_terakhir : { type: String, required: true },
        tahun_mulai_bertani : { type: Number, required: true },
        lahan_milik_sendiri : { type: Number, required: true },
        total_lahan_m2 : { type: Number, required: true },
        varietas_padi : { type: String, required: true },
        durasi_varietas_padi : { type: Number, required: true },
    },
    { timestamps: true }
);

const KuesionerManajemenSchema = new Schema(
    {
        id_petani: { type: String, required: true },
        kab_kota : { type: String, required: true },
        pembajakan_lahan_modern : { type: String, required: true },
        pengairan_sumur_bor : { type: String, required: true },
        pengairan_pompa_air : { type: String, required: true },
        penyemprotan_pompa_tangan : { type: String, required: true },
        penyemprotan_pompa_elektrik : { type: String, required: true },
        panen_mesin_potong : { type: String, required: true },
        pakai_internet : { type: String, required: true },
        info_penyuluh : { type: String, required: true },
        info_keuchik : { type: String, required: true },
        info_keujrun_blang : { type: String, required: true },
        anggota_kelompok_tani : { type: String, required: true },
        tahu_katam : { type: String, required: true },
        tahu_pergeseran_musim : { type: String, required: true },
        respon_pergeseran_musim : { type: String, required: true },
        pernah_gagal_tanam : { type: String, required: true },
        penyebab_gagal : { type: String, required: true },
        teknologi_lain : { type: String, required: true },
    },
    { timestamps: true }
);

export const KuesionerPetani = mongoose.models.KuesionerPetani || mongoose.model("KuesionerPetani", KuesionerPetaniSchema, "keusioner_petani");
export const KuesionerManajemen = mongoose.models.KuesionerManajemen || mongoose.model("KuesionerManajemen", KuesionerManajemenSchema, "kuesioner_manajemen");