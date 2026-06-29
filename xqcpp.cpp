#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/pytypes.h>

#include <vector>
#include <string>
#include <cstdint>
#include <stdexcept>
#include <algorithm>

namespace py = pybind11;

struct BitBoard90 {
    uint64_t lo; // bits 0..63
    uint64_t hi; // bits 64..89 used

    BitBoard90(uint64_t lo_=0, uint64_t hi_=0) : lo(lo_), hi(hi_) {}
};

static inline BitBoard90 bb_or(const BitBoard90& a, const BitBoard90& b) {
    return BitBoard90(a.lo | b.lo, a.hi | b.hi);
}
static inline BitBoard90 bb_and(const BitBoard90& a, const BitBoard90& b) {
    return BitBoard90(a.lo & b.lo, a.hi & b.hi);
}
static inline BitBoard90 bb_xor(const BitBoard90& a, const BitBoard90& b) {
    return BitBoard90(a.lo ^ b.lo, a.hi ^ b.hi);
}
static inline BitBoard90 bb_not(const BitBoard90& a) {
    // only keep 90 bits
    return BitBoard90(~a.lo, (~a.hi) & ((1ULL << 26) - 1));
}
static inline bool bb_any(const BitBoard90& a) {
    return a.lo != 0 || a.hi != 0;
}
static inline bool bb_eq(const BitBoard90& a, const BitBoard90& b) {
    return a.lo == b.lo && a.hi == b.hi;
}
static inline bool bb_has(const BitBoard90& a, const BitBoard90& b) {
    // any overlap
    return ((a.lo & b.lo) != 0) || ((a.hi & b.hi) != 0);
}
static inline void bb_ior(BitBoard90& a, const BitBoard90& b) {
    a.lo |= b.lo; a.hi |= b.hi;
}
static inline void bb_iand(BitBoard90& a, const BitBoard90& b) {
    a.lo &= b.lo; a.hi &= b.hi;
}
static inline void bb_ixor(BitBoard90& a, const BitBoard90& b) {
    a.lo ^= b.lo; a.hi ^= b.hi;
}

static inline BitBoard90 bb_from_square(int sq) {
    if (sq < 64) return BitBoard90(1ULL << sq, 0);
    return BitBoard90(0, 1ULL << (sq - 64));
}

static inline int msb_u64(uint64_t x) {
    unsigned long idx;
#if defined(_MSC_VER)
    _BitScanReverse64(&idx, x);
    return static_cast<int>(idx);
#else
    return 63 - __builtin_clzll(x);
#endif
}

static inline int bb_msb(const BitBoard90& a) {
    if (a.hi) return 64 + msb_u64(a.hi);
    return msb_u64(a.lo);
}

static inline BitBoard90 bb_without_square(const BitBoard90& a, int sq) {
    BitBoard90 m = bb_from_square(sq);
    return BitBoard90(a.lo & ~m.lo, a.hi & ~m.hi);
}

static BitBoard90 pyint_to_bb(const py::int_& obj) {
    py::object mask64 = py::int_((uint64_t)(~0ULL));
    py::int_ lo_py = py::reinterpret_steal<py::int_>(PyNumber_And(obj.ptr(), mask64.ptr()));
    uint64_t lo = lo_py.cast<uint64_t>();

    py::object shifted = py::reinterpret_steal<py::object>(PyNumber_Rshift(obj.ptr(), py::int_(64).ptr()));
    py::int_ hi_py = py::reinterpret_steal<py::int_>(PyNumber_And(shifted.ptr(), py::int_((1ULL << 26) - 1).ptr()));
    uint64_t hi = hi_py.cast<uint64_t>();
    return BitBoard90(lo, hi);
}

static py::int_ bb_to_pyint(const BitBoard90& a) {
    py::int_ lo(a.lo);
    py::int_ hi(a.hi);
    py::object shifted = py::reinterpret_steal<py::object>(PyNumber_Lshift(hi.ptr(), py::int_(64).ptr()));
    py::object total = py::reinterpret_steal<py::object>(PyNumber_Or(lo.ptr(), shifted.ptr()));
    return py::reinterpret_steal<py::int_>(total.release().ptr());
}

static const int N = 90;
static BitBoard90 BB[N];
static BitBoard90 BB_ALL;

static BitBoard90 PALACE[2], ADVISOR_POS[2], BISHOP_POS[2], PAWN_POS[2];
static BitBoard90 PAWN_ATTACKS[2][N];
static BitBoard90 PAWNS_CAN_ATTACK[2][N];
static BitBoard90 KING_ATTACKS[2][N];
static BitBoard90 ADVISOR_ATTACKS[2][N];

static inline int col(int s) { return s % 9; }
static inline int row(int s) { return s / 9; }

static inline int sqdist(int a, int b) {
    int dc = col(a) - col(b); if (dc < 0) dc = -dc;
    int dr = row(a) - row(b); if (dr < 0) dr = -dr;
    return dc > dr ? dc : dr;
}

static BitBoard90 sliding_attacks(int sq, const BitBoard90& occ, const int* deltas, int nd) {
    BitBoard90 a;
    for (int i = 0; i < nd; i++) {
        int d = deltas[i], s = sq;
        while (true) {
            int prev = s;
            s += d;
            if (s < 0 || s >= N || sqdist(s, prev) > 2) break;
            bb_ior(a, BB[s]);
            if (bb_has(occ, BB[s])) break;
        }
    }
    return a;
}

static BitBoard90 step_attacks(int sq, const int* d, int nd, const BitBoard90& restriction) {
    if (!bb_has(BB[sq], restriction)) return BitBoard90();
    BitBoard90 a = sliding_attacks(sq, BB_ALL, d, nd);
    bb_iand(a, restriction);
    return a;
}

static const int ROOK_DELTAS[4] = {1, -1, 9, -9};
static BitBoard90 rook_attacks(int sq, const BitBoard90& occ) {
    return sliding_attacks(sq, occ, ROOK_DELTAS, 4);
}

static const int KNIGHT_LEG[4] = {1, 9, -1, -9};
static const int KNIGHT_ATT[8] = {-7, 11, 17, 19, -11, 7, -19, -17};

static BitBoard90 knight_attacks(int sq, const BitBoard90& occ) {
    BitBoard90 a;
    for (int i = 0; i < 4; i++) {
        int leg = sq + KNIGHT_LEG[i];
        if (leg < 0 || leg >= N || bb_has(occ, BB[leg])) continue;
        for (int j = 0; j < 2; j++) {
            int s = sq + KNIGHT_ATT[2 * i + j];
            if (s < 0 || s >= N || sqdist(s, sq) > 2) continue;
            bb_ior(a, BB[s]);
        }
    }
    return a;
}

static const int KNIGHT_ATKLEG[4] = {8, 10, -8, -10};
static const int KNIGHT_ATKDEL[8] = {7, 17, 19, 11, -7, -17, -19, -11};

static BitBoard90 knights_can_attack(int sq, const BitBoard90& occ) {
    BitBoard90 a;
    for (int i = 0; i < 4; i++) {
        int leg = sq + KNIGHT_ATKLEG[i];
        if (leg < 0 || leg >= N || bb_has(occ, BB[leg])) continue;
        for (int j = 0; j < 2; j++) {
            int s = sq + KNIGHT_ATKDEL[2 * i + j];
            if (s < 0 || s >= N || sqdist(s, sq) > 2) continue;
            bb_ior(a, BB[s]);
        }
    }
    return a;
}

static const int BISHOP_EYE[4] = {8, -8, 10, -10};
static const int BISHOP_ATT[4] = {16, -16, 20, -20};

static BitBoard90 bishop_attacks(int sq, const BitBoard90& occ, int color) {
    BitBoard90 a;
    for (int i = 0; i < 4; i++) {
        int eye = sq + BISHOP_EYE[i];
        if (eye < 0 || eye >= N || bb_has(occ, BB[eye])) continue;
        int s = sq + BISHOP_ATT[i];
        if (s < 0 || s >= N || sqdist(s, sq) > 2) continue;
        bb_ior(a, BB[s]);
    }
    bb_iand(a, BISHOP_POS[color]);
    return a;
}

static BitBoard90 cannon_attacks(int sq, const BitBoard90& occ) {
    BitBoard90 a;
    for (int i = 0; i < 4; i++) {
        int d = ROOK_DELTAS[i], s = sq, cnt = 0;
        while (true) {
            int prev = s;
            s += d;
            if (s < 0 || s >= N || sqdist(s, prev) > 2) break;
            if (bb_has(occ, BB[s])) {
                cnt++;
                if (cnt == 2) {
                    bb_ior(a, BB[s]);
                    break;
                }
            }
        }
    }
    return a;
}

static BitBoard90 cannon_slides(int sq, const BitBoard90& occ) {
    BitBoard90 a;
    for (int i = 0; i < 4; i++) {
        int d = ROOK_DELTAS[i], s = sq;
        while (true) {
            int prev = s;
            s += d;
            if (s < 0 || s >= N || sqdist(s, prev) > 2) break;
            if (bb_has(occ, BB[s])) break;
            bb_ior(a, BB[s]);
        }
    }
    return a;
}

struct State {
    BitBoard90 pawns, rooks, knights, bishops, advisors, kings, cannons;
    BitBoard90 occ0, occ1, occupied;
    bool turn;

    inline const BitBoard90& occ(int c) const { return c ? occ1 : occ0; }
};

static BitBoard90 attacks_mask(const State& s, int sq) {
    BitBoard90 m = BB[sq];
    if (bb_has(m, s.pawns)) {
        int c = bb_has(m, s.occ1) ? 1 : 0;
        return PAWN_ATTACKS[c][sq];
    }
    if (bb_has(m, s.rooks)) return rook_attacks(sq, s.occupied);
    if (bb_has(m, s.knights)) return knight_attacks(sq, s.occupied);
    if (bb_has(m, s.bishops)) {
        int c = bb_has(m, s.occ1) ? 1 : 0;
        return bishop_attacks(sq, s.occupied, c);
    }
    if (bb_has(m, s.advisors)) {
        int c = bb_has(m, s.occ1) ? 1 : 0;
        return ADVISOR_ATTACKS[c][sq];
    }
    if (bb_has(m, s.kings)) {
        int c = bb_has(m, s.occ1) ? 1 : 0;
        return KING_ATTACKS[c][sq];
    }
    if (bb_has(m, s.cannons)) return cannon_attacks(sq, s.occupied);
    return BitBoard90();
}

static BitBoard90 attackers_mask(const State& s, int color, int sq, const BitBoard90& occ) {
    BitBoard90 att;
    bb_ior(att, bb_and(rook_attacks(sq, occ), s.rooks));
    bb_ior(att, bb_and(knights_can_attack(sq, occ), s.knights));
    bb_ior(att, bb_and(bishop_attacks(sq, occ, color), s.bishops));
    bb_ior(att, bb_and(ADVISOR_ATTACKS[color][sq], s.advisors));
    bb_ior(att, bb_and(KING_ATTACKS[color][sq], s.kings));
    bb_ior(att, bb_and(PAWNS_CAN_ATTACK[color][sq], s.pawns));
    bb_ior(att, bb_and(cannon_attacks(sq, occ), s.cannons));
    bb_iand(att, s.occ(color));
    return att;
}

static int king_sq(const State& s, int color) {
    BitBoard90 km = bb_and(s.occ(color), s.kings);
    return bb_any(km) ? bb_msb(km) : -1;
}

static bool kings_face(const State& s) {
    int rk = king_sq(s, 1), bk = king_sq(s, 0);
    if (rk < 0 || bk < 0) return false;

    if (col(rk) == col(bk)) {
        int lo = std::min(rk, bk), hi = std::max(rk, bk);
        bool saw = false;
        for (int t = lo + 9; t < hi; t += 9) {
            saw = true;
            if (bb_has(s.occupied, BB[t])) return false;
        }
        return saw;
    }
    if (row(rk) == row(bk)) {
        int lo = std::min(rk, bk), hi = std::max(rk, bk);
        bool saw = false;
        for (int t = lo + 1; t < hi; t++) {
            saw = true;
            if (bb_has(s.occupied, BB[t])) return false;
        }
        return saw;
    }
    return false;
}

static void clear_square(State& s, int sq) {
    BitBoard90 m = BB[sq];
    if (!bb_has(s.occupied, m)) return;

    s.pawns    = bb_and(s.pawns,    bb_not(m));
    s.rooks    = bb_and(s.rooks,    bb_not(m));
    s.knights  = bb_and(s.knights,  bb_not(m));
    s.bishops  = bb_and(s.bishops,  bb_not(m));
    s.advisors = bb_and(s.advisors, bb_not(m));
    s.kings    = bb_and(s.kings,    bb_not(m));
    s.cannons  = bb_and(s.cannons,  bb_not(m));
    s.occ0     = bb_and(s.occ0,     bb_not(m));
    s.occ1     = bb_and(s.occ1,     bb_not(m));
    s.occupied = bb_and(s.occupied, bb_not(m));
}

static void move_piece(State& s, int from, int to) {
    BitBoard90 fm = BB[from], tm = BB[to];
    int c = bb_has(s.occ1, fm) ? 1 : 0;

    if (bb_has(s.pawns, fm))      { s.pawns = bb_and(s.pawns, bb_not(fm)); bb_ior(s.pawns, tm); }
    else if (bb_has(s.rooks, fm)) { s.rooks = bb_and(s.rooks, bb_not(fm)); bb_ior(s.rooks, tm); }
    else if (bb_has(s.knights, fm)){ s.knights = bb_and(s.knights, bb_not(fm)); bb_ior(s.knights, tm); }
    else if (bb_has(s.bishops, fm)){ s.bishops = bb_and(s.bishops, bb_not(fm)); bb_ior(s.bishops, tm); }
    else if (bb_has(s.advisors, fm)){ s.advisors = bb_and(s.advisors, bb_not(fm)); bb_ior(s.advisors, tm); }
    else if (bb_has(s.kings, fm)) { s.kings = bb_and(s.kings, bb_not(fm)); bb_ior(s.kings, tm); }
    else if (bb_has(s.cannons, fm)){ s.cannons = bb_and(s.cannons, bb_not(fm)); bb_ior(s.cannons, tm); }

    if (c) {
        s.occ1 = bb_and(s.occ1, bb_not(fm));
        bb_ior(s.occ1, tm);
    } else {
        s.occ0 = bb_and(s.occ0, bb_not(fm));
        bb_ior(s.occ0, tm);
    }
    s.occupied = bb_and(s.occupied, bb_not(fm));
    bb_ior(s.occupied, tm);
}

static bool is_safe(const State& s, int from, int to) {
    State s2 = s;
    clear_square(s2, to);
    move_piece(s2, from, to);

    int mover = s.turn ? 1 : 0;
    int ks = king_sq(s2, mover);
    if (ks >= 0 && bb_any(attackers_mask(s2, mover ^ 1, ks, s2.occupied))) return false;
    if (kings_face(s2)) return false;
    return true;
}

static inline void push_move(std::vector<std::string>& out, int from, int to) {
    char b[4];
    b[0] = 'a' + col(from);
    b[1] = '0' + row(from);
    b[2] = 'a' + col(to);
    b[3] = '0' + row(to);
    out.emplace_back(b, 4);
}

static std::vector<std::string> gen_legal(const State& s) {
    std::vector<std::string> out;
    out.reserve(64);

    int mk = king_sq(s, s.turn ? 1 : 0);
    int ok = king_sq(s, s.turn ? 0 : 1);
    BitBoard90 our = s.occ(s.turn ? 1 : 0);

    BitBoard90 fset = our;
    while (bb_any(fset)) {
        int from = bb_msb(fset);
        fset = bb_without_square(fset, from);

        BitBoard90 ms = attacks_mask(s, from);
        ms = bb_and(ms, bb_not(our));

        while (bb_any(ms)) {
            int to = bb_msb(ms);
            ms = bb_without_square(ms, to);

            if (mk < 0) push_move(out, from, to);
            else if (to == ok) push_move(out, from, to);
            else if (is_safe(s, from, to)) push_move(out, from, to);
        }

        if (bb_has(s.cannons, BB[from])) {
            BitBoard90 ss = cannon_slides(from, s.occupied);
            while (bb_any(ss)) {
                int to = bb_msb(ss);
                ss = bb_without_square(ss, to);

                if (mk < 0) push_move(out, from, to);
                else if (to == ok) push_move(out, from, to);
                else if (is_safe(s, from, to)) push_move(out, from, to);
            }
        }
    }
    return out;
}

static void init_tables() {
    for (int s = 0; s < N; s++) BB[s] = bb_from_square(s);

    // 90 bits all ones
    BB_ALL = BitBoard90(~0ULL, (1ULL << 26) - 1);

    PALACE[0] = PALACE[1] = BitBoard90();
    for (int r = 0; r <= 2; r++) for (int c = 3; c <= 5; c++) bb_ior(PALACE[1], BB[r * 9 + c]);
    for (int r = 7; r <= 9; r++) for (int c = 3; c <= 5; c++) bb_ior(PALACE[0], BB[r * 9 + c]);

    {
        int red[5] = {3, 5, 13, 21, 23};
        int blk[5] = {84, 86, 76, 68, 66};
        ADVISOR_POS[1] = ADVISOR_POS[0] = BitBoard90();
        for (int i = 0; i < 5; i++) {
            bb_ior(ADVISOR_POS[1], BB[red[i]]);
            bb_ior(ADVISOR_POS[0], BB[blk[i]]);
        }
    }

    {
        int red[7] = {2, 6, 18, 22, 26, 38, 42};
        int blk[7] = {83, 87, 63, 67, 71, 47, 51};
        BISHOP_POS[1] = BISHOP_POS[0] = BitBoard90();
        for (int i = 0; i < 7; i++) {
            bb_ior(BISHOP_POS[1], BB[red[i]]);
            bb_ior(BISHOP_POS[0], BB[blk[i]]);
        }
    }

    PAWN_POS[0] = PAWN_POS[1] = BitBoard90();
    for (int s = 0; s < N; s++) {
        int r = row(s);
        bool ev = (col(s) % 2 == 0);
        if (r >= 5 || ((r == 3 || r == 4) && ev)) bb_ior(PAWN_POS[1], BB[s]); // red
        if (r <= 4 || ((r == 5 || r == 6) && ev)) bb_ior(PAWN_POS[0], BB[s]); // black
    }

    int dn[1] = {-9}, up[1] = {9}, dns[3] = {-9, -1, 1}, ups[3] = {9, -1, 1};
    int kd[4] = {9, -9, 1, -1}, ad[4] = {8, -8, 10, -10};

    for (int s = 0; s < N; s++) {
        if (s < 45) {
            PAWN_ATTACKS[0][s] = step_attacks(s, dns, 3, PAWN_POS[0]);
            PAWN_ATTACKS[1][s] = step_attacks(s, up, 1, PAWN_POS[1]);
            PAWNS_CAN_ATTACK[0][s] = step_attacks(s, ups, 3, PAWN_POS[0]);
            PAWNS_CAN_ATTACK[1][s] = step_attacks(s, dn, 1, PAWN_POS[1]);
        } else {
            PAWN_ATTACKS[0][s] = step_attacks(s, dn, 1, PAWN_POS[0]);
            PAWN_ATTACKS[1][s] = step_attacks(s, ups, 3, PAWN_POS[1]);
            PAWNS_CAN_ATTACK[0][s] = step_attacks(s, up, 1, PAWN_POS[0]);
            PAWNS_CAN_ATTACK[1][s] = step_attacks(s, dns, 3, PAWN_POS[1]);
        }
        KING_ATTACKS[0][s] = step_attacks(s, kd, 4, PALACE[0]);
        KING_ATTACKS[1][s] = step_attacks(s, kd, 4, PALACE[1]);
        ADVISOR_ATTACKS[0][s] = step_attacks(s, ad, 4, ADVISOR_POS[0]);
        ADVISOR_ATTACKS[1][s] = step_attacks(s, ad, 4, ADVISOR_POS[1]);
    }
}

static std::vector<std::string> legal_moves_uci(
    py::int_ pawns,
    py::int_ rooks,
    py::int_ knights,
    py::int_ bishops,
    py::int_ advisors,
    py::int_ kings,
    py::int_ cannons,
    py::int_ occ_red,
    py::int_ occ_black,
    bool turn
) {
    State s;
    s.pawns = pyint_to_bb(pawns);
    s.rooks = pyint_to_bb(rooks);
    s.knights = pyint_to_bb(knights);
    s.bishops = pyint_to_bb(bishops);
    s.advisors = pyint_to_bb(advisors);
    s.kings = pyint_to_bb(kings);
    s.cannons = pyint_to_bb(cannons);
    s.occ1 = pyint_to_bb(occ_red);
    s.occ0 = pyint_to_bb(occ_black);
    s.occupied = bb_or(s.occ1, s.occ0);
    s.turn = turn;
    return gen_legal(s);
}

static py::tuple gen_legal_and_in_check(
    py::int_ pawns, py::int_ rooks, py::int_ knights, py::int_ bishops,
    py::int_ advisors, py::int_ kings, py::int_ cannons,
    py::int_ occ_red, py::int_ occ_black, bool turn
) {
    State s;
    s.pawns = pyint_to_bb(pawns);     s.rooks = pyint_to_bb(rooks);
    s.knights = pyint_to_bb(knights); s.bishops = pyint_to_bb(bishops);
    s.advisors = pyint_to_bb(advisors); s.kings = pyint_to_bb(kings);
    s.cannons = pyint_to_bb(cannons);
    s.occ1 = pyint_to_bb(occ_red);    s.occ0 = pyint_to_bb(occ_black);
    s.occupied = bb_or(s.occ1, s.occ0); s.turn = turn;

    bool chk = false;
    int mk = king_sq(s, s.turn ? 1 : 0);
    if (mk >= 0)
        chk = bb_any(attackers_mask(s, s.turn ? 0 : 1, mk, s.occupied));

    std::vector<std::string> moves = gen_legal(s);
    return py::make_tuple(moves, chk);
}

PYBIND11_MODULE(xqcpp, m) {
    init_tables();
    m.def("legal_moves_uci", &legal_moves_uci,
        py::arg("pawns"),
        py::arg("rooks"),
        py::arg("knights"),
        py::arg("bishops"),
        py::arg("advisors"),
        py::arg("kings"),
        py::arg("cannons"),
        py::arg("occ_red"),
        py::arg("occ_black"),
        py::arg("turn")
    );
    m.def("gen_legal_and_in_check", &gen_legal_and_in_check,
        py::arg("pawns"),
        py::arg("rooks"),
        py::arg("knights"),
        py::arg("bishops"),
        py::arg("advisors"),
        py::arg("kings"),
        py::arg("cannons"),
        py::arg("occ_red"),
        py::arg("occ_black"),
        py::arg("turn")
    );
}