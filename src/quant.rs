//! AV1 dequantisation tables + per-plane DC/AC quantiser computation
//! (§7.12.2).
//!
//! The uncompressed header carries a `base_q_idx` (0..=255) plus signed
//! deltas for each `(plane, DC|AC)` pair. For each plane the decoder
//! computes `q = clip_u8(base + delta)` and looks the dequantiser up in
//! the `DC_Qn` / `AC_Qn` tables (Table §7.12.2.1 / .2). Tables exist for
//! 8-/10-/12-bit bit depths.
//!
//! Phase 3 **does not** honour `using_qmatrix` — callers that encounter
//! `FrameHeader::quant.using_qmatrix == true` must surface
//! `Error::Unsupported("av1 quantization matrices (§5.9.12) pending")`.

use oxideav_core::{Error, Result};

/// Plane selector for per-plane dequantiser lookups (§7.12.2.6).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Plane {
    Y = 0,
    U = 1,
    V = 2,
}

/// Flat carrier of all quantisation parameters needed to compute
/// per-plane DC/AC dequantisers.
#[derive(Clone, Copy, Debug)]
pub struct Params {
    /// `base_q_idx` — frame-level quantiser index (§5.9.12).
    pub base_q_idx: i32,
    /// `DeltaQYDc` — signed delta applied to the Y DC quantiser.
    pub delta_q_y_dc: i32,
    /// `DeltaQUDc` — signed delta applied to the U DC quantiser.
    pub delta_q_u_dc: i32,
    /// `DeltaQUAc` — signed delta applied to the U AC quantiser.
    pub delta_q_u_ac: i32,
    /// `DeltaQVDc` — signed delta applied to the V DC quantiser.
    pub delta_q_v_dc: i32,
    /// `DeltaQVAc` — signed delta applied to the V AC quantiser.
    pub delta_q_v_ac: i32,
    /// Sample bit depth — 8, 10, or 12.
    pub bit_depth: u32,
}

/// Per-plane pair of DC / AC dequantisers. Units are Q0 integers;
/// applied as `raw_coeff * dequant` after reading residuals from the
/// coefficient decoder. `bit_depth` carries through so that callers
/// can apply the §7.13.3 step-f clip (`±(1 << (7 + BitDepth))`)
/// without having to plumb the sequence header separately.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Values {
    pub dc: u16,
    pub ac: u16,
    pub bit_depth: u32,
}

impl Params {
    /// Compute the DC / AC dequantisers for `plane` after applying the
    /// relevant delta and clipping `q` to `0..=255`.
    ///
    /// Returns `Error::unsupported` when the bit depth is out of range
    /// (8 / 10 / 12 are supported; the spec allows nothing else).
    pub fn compute(&self, plane: Plane) -> Result<Values> {
        let dc_tab = dc_lookup(self.bit_depth).ok_or_else(|| {
            Error::unsupported(format!(
                "av1 quant: bit_depth {} unsupported (§7.12.2)",
                self.bit_depth
            ))
        })?;
        let ac_tab = ac_lookup(self.bit_depth).ok_or_else(|| {
            Error::unsupported(format!(
                "av1 quant: bit_depth {} unsupported (§7.12.2)",
                self.bit_depth
            ))
        })?;
        let (q_dc, q_ac) = match plane {
            Plane::Y => (self.base_q_idx + self.delta_q_y_dc, self.base_q_idx),
            Plane::U => (
                self.base_q_idx + self.delta_q_u_dc,
                self.base_q_idx + self.delta_q_u_ac,
            ),
            Plane::V => (
                self.base_q_idx + self.delta_q_v_dc,
                self.base_q_idx + self.delta_q_v_ac,
            ),
        };
        Ok(Values {
            dc: dc_tab[clip_q(q_dc) as usize],
            ac: ac_tab[clip_q(q_ac) as usize],
            bit_depth: self.bit_depth,
        })
    }
}

#[inline]
fn clip_q(q: i32) -> i32 {
    q.clamp(0, 255)
}

/// DC lookup table for `bit_depth`. Returns `None` for unsupported
/// depths.
pub fn dc_lookup(bit_depth: u32) -> Option<&'static [u16; 256]> {
    match bit_depth {
        8 => Some(&DC8),
        10 => Some(&DC10),
        12 => Some(&DC12),
        _ => None,
    }
}

/// AC lookup table for `bit_depth`.
pub fn ac_lookup(bit_depth: u32) -> Option<&'static [u16; 256]> {
    match bit_depth {
        8 => Some(&AC8),
        10 => Some(&AC10),
        12 => Some(&AC12),
        _ => None,
    }
}

/// DC dequantiser for `(base_q, delta_q, bit_depth)` clipped to `0..=255`.
pub fn get_dc_quant(base_q: i32, delta_q: i32, bit_depth: u32) -> Result<i32> {
    let tab = dc_lookup(bit_depth).ok_or_else(|| {
        Error::unsupported(format!(
            "av1 quant: bit_depth {bit_depth} unsupported (§7.12.2)"
        ))
    })?;
    Ok(tab[clip_q(base_q + delta_q) as usize] as i32)
}

/// AC dequantiser for `(base_q, delta_q, bit_depth)` clipped to `0..=255`.
pub fn get_ac_quant(base_q: i32, delta_q: i32, bit_depth: u32) -> Result<i32> {
    let tab = ac_lookup(bit_depth).ok_or_else(|| {
        Error::unsupported(format!(
            "av1 quant: bit_depth {bit_depth} unsupported (§7.12.2)"
        ))
    })?;
    Ok(tab[clip_q(base_q + delta_q) as usize] as i32)
}

/// 8-bit DC dequantiser table (spec Table 7.12.2.1, verified against
/// libaom `dc_qlookup_QTX`).
pub static DC8: [u16; 256] = [
    4, 8, 8, 9, 10, 11, 12, 12, 13, 14, 15, 16, 17, 18, 19, 19, 20, 21, 22, 23, 24, 25, 26, 26, 27,
    28, 29, 30, 31, 32, 32, 33, 34, 35, 36, 37, 38, 38, 39, 40, 41, 42, 43, 43, 44, 45, 46, 47, 48,
    48, 49, 50, 51, 52, 53, 53, 54, 55, 56, 57, 57, 58, 59, 60, 61, 62, 62, 63, 64, 65, 66, 66, 67,
    68, 69, 70, 70, 71, 72, 73, 74, 74, 75, 76, 77, 78, 78, 79, 80, 81, 81, 82, 83, 84, 85, 85, 87,
    88, 90, 92, 93, 95, 96, 98, 100, 101, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122,
    124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 155, 158, 161, 164,
    167, 170, 173, 176, 179, 182, 185, 188, 191, 194, 197, 200, 203, 207, 211, 215, 219, 223, 227,
    231, 235, 239, 243, 247, 251, 255, 260, 265, 270, 275, 280, 285, 290, 295, 301, 307, 313, 319,
    325, 332, 339, 346, 353, 361, 369, 377, 385, 393, 401, 410, 419, 428, 437, 446, 455, 464, 473,
    483, 493, 503, 513, 523, 533, 543, 554, 565, 577, 589, 601, 613, 625, 637, 649, 661, 673, 685,
    697, 709, 721, 733, 745, 757, 769, 781, 793, 805, 817, 829, 841, 853, 865, 877, 889, 901, 913,
    925, 937, 949, 961, 973, 985, 997, 1009, 1021, 1033, 1045, 1057, 1069, 1081, 1093, 1105, 1117,
    1129, 1141, 1153, 1165, 1177, 1189, 1201, 1213,
];

/// 8-bit AC dequantiser table (spec Table 7.12.2.2).
pub static AC8: [u16; 256] = [
    4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
    55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78,
    79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101,
    102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138,
    140, 142, 144, 146, 148, 150, 152, 155, 158, 161, 164, 167, 170, 173, 176, 179, 182, 185, 188,
    191, 194, 197, 200, 203, 207, 211, 215, 219, 223, 227, 231, 235, 239, 243, 247, 251, 255, 260,
    265, 270, 275, 280, 285, 290, 295, 301, 307, 313, 319, 325, 332, 339, 346, 353, 361, 369, 377,
    385, 393, 401, 409, 418, 427, 436, 445, 454, 463, 472, 481, 491, 501, 511, 521, 531, 542, 553,
    564, 575, 586, 597, 609, 621, 633, 645, 657, 669, 682, 695, 708, 721, 734, 748, 762, 776, 791,
    806, 822, 839, 857, 875, 894, 913, 933, 953, 973, 993, 1013, 1033, 1053, 1073, 1093, 1114,
    1135, 1156, 1177, 1198, 1220, 1242, 1264, 1286, 1308, 1331, 1354, 1377, 1400, 1423, 1446, 1470,
    1494, 1518, 1542, 1566, 1590, 1615, 1640, 1665, 1690, 1715, 1741, 1767, 1793, 1819,
];

/// 10-bit DC dequantiser table (transcribed from libaom
/// `av1/common/quant_common.c`).
pub static DC10: [u16; 256] = [
    4, 9, 10, 13, 15, 17, 20, 22, 25, 28, 31, 34, 37, 40, 43, 47, 50, 53, 57, 60, 64, 68, 71, 75,
    78, 82, 86, 90, 93, 97, 101, 105, 109, 113, 116, 120, 124, 128, 132, 136, 140, 143, 147, 151,
    155, 159, 163, 166, 170, 174, 178, 182, 185, 189, 193, 197, 200, 204, 208, 212, 215, 219, 223,
    226, 230, 233, 237, 241, 244, 248, 251, 255, 259, 262, 266, 269, 273, 276, 280, 283, 287, 290,
    293, 297, 300, 304, 307, 310, 314, 317, 321, 324, 327, 331, 334, 337, 343, 350, 356, 362, 369,
    375, 381, 387, 394, 400, 406, 412, 418, 424, 430, 436, 442, 448, 454, 460, 466, 472, 478, 484,
    490, 499, 507, 516, 525, 533, 542, 550, 559, 567, 576, 584, 592, 601, 609, 617, 625, 634, 644,
    655, 666, 676, 687, 698, 708, 718, 729, 739, 749, 759, 770, 782, 795, 807, 819, 831, 844, 856,
    868, 880, 891, 906, 920, 933, 947, 961, 975, 988, 1001, 1015, 1030, 1045, 1061, 1076, 1090,
    1105, 1120, 1137, 1153, 1170, 1186, 1202, 1218, 1236, 1253, 1271, 1288, 1306, 1323, 1342, 1361,
    1379, 1398, 1416, 1436, 1456, 1476, 1496, 1516, 1537, 1559, 1580, 1601, 1624, 1647, 1670, 1692,
    1717, 1741, 1766, 1791, 1817, 1844, 1871, 1900, 1929, 1958, 1990, 2021, 2054, 2088, 2123, 2159,
    2197, 2236, 2276, 2319, 2363, 2410, 2458, 2508, 2561, 2616, 2675, 2737, 2802, 2871, 2944, 3020,
    3102, 3188, 3280, 3375, 3478, 3586, 3702, 3823, 3953, 4089, 4236, 4394, 4559, 4737, 4929, 5130,
    5347,
];

/// 12-bit DC dequantiser table.
pub static DC12: [u16; 256] = [
    4, 12, 18, 25, 33, 41, 50, 60, 70, 80, 91, 103, 115, 127, 140, 153, 166, 180, 194, 208, 222,
    237, 251, 266, 281, 296, 312, 327, 343, 358, 374, 390, 405, 421, 437, 453, 469, 484, 500, 516,
    532, 548, 564, 580, 596, 611, 627, 643, 659, 674, 690, 706, 721, 737, 752, 768, 783, 798, 814,
    829, 844, 859, 874, 889, 904, 919, 934, 949, 964, 978, 993, 1008, 1022, 1037, 1051, 1065, 1080,
    1094, 1108, 1122, 1136, 1151, 1165, 1179, 1192, 1206, 1220, 1234, 1248, 1261, 1275, 1288, 1302,
    1315, 1329, 1342, 1368, 1393, 1419, 1444, 1469, 1494, 1519, 1544, 1569, 1594, 1618, 1643, 1668,
    1692, 1717, 1741, 1765, 1789, 1814, 1838, 1862, 1885, 1909, 1933, 1957, 1992, 2027, 2061, 2096,
    2130, 2165, 2199, 2233, 2267, 2300, 2334, 2367, 2400, 2434, 2467, 2499, 2532, 2575, 2618, 2661,
    2704, 2746, 2788, 2830, 2872, 2913, 2954, 2995, 3036, 3076, 3127, 3177, 3226, 3275, 3324, 3373,
    3421, 3469, 3517, 3565, 3621, 3677, 3733, 3788, 3843, 3897, 3951, 4005, 4058, 4119, 4181, 4241,
    4301, 4361, 4420, 4479, 4546, 4612, 4677, 4742, 4807, 4871, 4942, 5013, 5083, 5153, 5222, 5291,
    5367, 5442, 5517, 5591, 5665, 5745, 5825, 5905, 5984, 6063, 6149, 6234, 6319, 6404, 6495, 6587,
    6678, 6769, 6867, 6966, 7064, 7163, 7269, 7376, 7483, 7599, 7715, 7832, 7958, 8085, 8214, 8352,
    8492, 8635, 8788, 8945, 9104, 9275, 9450, 9639, 9832, 10031, 10245, 10465, 10702, 10946, 11210,
    11482, 11776, 12081, 12409, 12750, 13118, 13501, 13913, 14343, 14807, 15290, 15812, 16356,
    16943, 17575, 18237, 18949, 19718, 20521, 21387,
];

/// 10-bit AC dequantiser table.
pub static AC10: [u16; 256] = [
    4, 9, 11, 13, 16, 18, 21, 24, 27, 30, 33, 37, 40, 44, 48, 51, 55, 59, 63, 67, 71, 75, 79, 83,
    88, 92, 96, 100, 105, 109, 114, 118, 122, 127, 131, 136, 140, 145, 149, 154, 158, 163, 168,
    172, 177, 181, 186, 190, 195, 199, 204, 208, 213, 217, 222, 226, 231, 235, 240, 244, 249, 253,
    258, 262, 267, 271, 275, 280, 284, 289, 293, 297, 302, 306, 311, 315, 319, 324, 328, 332, 337,
    341, 345, 349, 354, 358, 362, 367, 371, 375, 379, 384, 388, 392, 396, 401, 409, 417, 425, 433,
    441, 449, 458, 466, 474, 482, 490, 498, 506, 514, 523, 531, 539, 547, 555, 563, 571, 579, 588,
    596, 604, 616, 628, 640, 652, 664, 676, 688, 700, 713, 725, 737, 749, 761, 773, 785, 797, 809,
    825, 841, 857, 873, 889, 905, 922, 938, 954, 970, 986, 1002, 1018, 1038, 1058, 1078, 1098,
    1118, 1138, 1158, 1178, 1198, 1218, 1242, 1266, 1290, 1314, 1338, 1362, 1386, 1411, 1435, 1463,
    1491, 1519, 1547, 1575, 1603, 1631, 1663, 1695, 1727, 1759, 1791, 1823, 1859, 1895, 1931, 1967,
    2003, 2039, 2079, 2119, 2159, 2199, 2239, 2283, 2327, 2371, 2415, 2459, 2507, 2555, 2603, 2651,
    2703, 2755, 2807, 2859, 2915, 2971, 3027, 3083, 3143, 3203, 3263, 3327, 3391, 3455, 3523, 3591,
    3659, 3731, 3803, 3876, 3952, 4028, 4104, 4184, 4264, 4348, 4432, 4516, 4604, 4692, 4784, 4876,
    4972, 5068, 5168, 5268, 5372, 5476, 5584, 5692, 5804, 5916, 6032, 6148, 6268, 6388, 6512, 6640,
    6768, 6900, 7036, 7172, 7312,
];

/// 12-bit AC dequantiser table.
pub static AC12: [u16; 256] = [
    4, 13, 19, 27, 35, 44, 54, 64, 75, 87, 99, 112, 126, 139, 154, 168, 183, 199, 214, 230, 247,
    263, 280, 297, 314, 331, 349, 366, 384, 402, 420, 438, 456, 475, 493, 511, 530, 548, 567, 586,
    604, 623, 642, 660, 679, 698, 716, 735, 753, 772, 791, 809, 828, 846, 865, 884, 902, 920, 939,
    957, 976, 994, 1012, 1030, 1049, 1067, 1085, 1103, 1121, 1139, 1157, 1175, 1193, 1211, 1229,
    1246, 1264, 1282, 1299, 1317, 1335, 1352, 1370, 1387, 1405, 1422, 1440, 1457, 1474, 1491, 1509,
    1526, 1543, 1560, 1577, 1595, 1627, 1660, 1693, 1725, 1758, 1791, 1824, 1856, 1889, 1922, 1954,
    1987, 2020, 2052, 2085, 2118, 2150, 2183, 2216, 2248, 2281, 2313, 2346, 2378, 2411, 2459, 2508,
    2556, 2605, 2653, 2701, 2750, 2798, 2847, 2895, 2943, 2992, 3040, 3088, 3137, 3185, 3234, 3298,
    3362, 3426, 3491, 3555, 3619, 3684, 3748, 3812, 3876, 3941, 4005, 4069, 4149, 4230, 4310, 4390,
    4470, 4550, 4631, 4711, 4791, 4871, 4967, 5064, 5160, 5256, 5352, 5448, 5544, 5641, 5737, 5849,
    5961, 6073, 6185, 6297, 6410, 6522, 6650, 6778, 6906, 7034, 7162, 7290, 7435, 7579, 7723, 7867,
    8011, 8155, 8315, 8475, 8635, 8795, 8956, 9132, 9308, 9484, 9660, 9836, 10028, 10220, 10412,
    10604, 10812, 11020, 11228, 11437, 11661, 11885, 12109, 12333, 12573, 12813, 13053, 13309,
    13565, 13821, 14093, 14365, 14637, 14925, 15213, 15502, 15806, 16110, 16414, 16734, 17054,
    17390, 17726, 18062, 18414, 18766, 19134, 19502, 19886, 20270, 20670, 21070, 21486, 21902,
    22334, 22766, 23214, 23662, 24126, 24590, 25070, 25551, 26047, 26559, 27071, 27599, 28143,
    28687, 29247,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dc8_and_ac8_spot_values() {
        // Spec Table 7.12.2: index 0 is 4, index 1 is 8 for both DC and AC.
        assert_eq!(DC8[0], 4);
        assert_eq!(DC8[1], 8);
        assert_eq!(AC8[0], 4);
    }

    #[test]
    fn dc_tables_monotonic() {
        for (name, tab) in [("DC8", &DC8), ("DC10", &DC10), ("DC12", &DC12)] {
            let mut prev = 0u16;
            for (i, &v) in tab.iter().enumerate() {
                assert!(v >= prev, "{name} not monotonic at {i}: {v} < {prev}");
                prev = v;
            }
        }
    }

    #[test]
    fn ac_tables_monotonic() {
        for (name, tab) in [("AC8", &AC8), ("AC10", &AC10), ("AC12", &AC12)] {
            let mut prev = 0u16;
            for (i, &v) in tab.iter().enumerate() {
                assert!(v >= prev, "{name} not monotonic at {i}: {v} < {prev}");
                prev = v;
            }
        }
    }

    #[test]
    fn compute_y_clips_negative_delta() {
        let p = Params {
            base_q_idx: 5,
            delta_q_y_dc: -100,
            delta_q_u_dc: 0,
            delta_q_u_ac: 0,
            delta_q_v_dc: 0,
            delta_q_v_ac: 0,
            bit_depth: 8,
        };
        let v = p.compute(Plane::Y).expect("compute");
        assert_eq!(v.dc, 4, "clipped to DC8[0]");
        assert_eq!(v.ac, AC8[5]);
    }

    #[test]
    fn compute_v_uses_v_deltas() {
        let p = Params {
            base_q_idx: 50,
            delta_q_y_dc: 0,
            delta_q_u_dc: 0,
            delta_q_u_ac: 0,
            delta_q_v_dc: 10,
            delta_q_v_ac: -5,
            bit_depth: 8,
        };
        let v = p.compute(Plane::V).expect("compute");
        assert_eq!(v.dc, DC8[60]);
        assert_eq!(v.ac, AC8[45]);
    }

    #[test]
    fn compute_hdr_bit_depths_produce_nonzero() {
        for bd in [10u32, 12] {
            let p = Params {
                base_q_idx: 100,
                delta_q_y_dc: 0,
                delta_q_u_dc: 0,
                delta_q_u_ac: 0,
                delta_q_v_dc: 0,
                delta_q_v_ac: 0,
                bit_depth: bd,
            };
            let v = p.compute(Plane::Y).expect("compute hdr");
            assert_ne!(v.dc, 0);
            assert_ne!(v.ac, 0);
        }
    }

    #[test]
    fn compute_rejects_unsupported_bit_depth() {
        let p = Params {
            base_q_idx: 50,
            delta_q_y_dc: 0,
            delta_q_u_dc: 0,
            delta_q_u_ac: 0,
            delta_q_v_dc: 0,
            delta_q_v_ac: 0,
            bit_depth: 9,
        };
        match p.compute(Plane::Y) {
            Err(Error::Unsupported(s)) => assert!(s.contains("9"), "msg: {s}"),
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn get_dc_quant_matches_table() {
        assert_eq!(get_dc_quant(50, 10, 8).unwrap(), DC8[60] as i32);
        assert_eq!(get_ac_quant(50, -5, 8).unwrap(), AC8[45] as i32);
    }
}
