"""Seed comprehensive 2026/27 Süper Lig squads, player ratings, and unit synergies into DuckDB."""

from datetime import datetime
import json
from pathlib import Path
from typing import Any
import duckdb

SQUADS_DATA: dict[int, dict[str, Any]] = {
    1: {  # Galatasaray
        "name": "Galatasaray",
        "players": [
            {"name": "Ugurcan Cakir", "pos": "GK", "nat": "TR", "age": 30, "att": 0.00, "def": 0.38, "ovr": 83, "starter": True, "val": 28000000},
            {"name": "Gunay Guvenc", "pos": "GK", "nat": "TR", "age": 35, "att": 0.00, "def": 0.12, "ovr": 72, "starter": False, "val": 500000},
            {"name": "Fernando Muslera", "pos": "GK", "nat": "UY", "age": 40, "att": 0.00, "def": 0.22, "ovr": 76, "starter": False, "val": 800000},
            {"name": "Davinson Sanchez", "pos": "CB", "nat": "CO", "age": 30, "att": 0.06, "def": 0.38, "ovr": 83, "starter": True, "val": 20000000},
            {"name": "Abdulkerim Bardakci", "pos": "CB", "nat": "TR", "age": 32, "att": 0.08, "def": 0.30, "ovr": 79, "starter": True, "val": 7000000},
            {"name": "Victor Nelsson", "pos": "CB", "nat": "DK", "age": 28, "att": 0.03, "def": 0.26, "ovr": 78, "starter": False, "val": 14000000},
            {"name": "Metehan Baltaci", "pos": "CB", "nat": "TR", "age": 23, "att": 0.03, "def": 0.24, "ovr": 76, "starter": False, "val": 4000000},
            {"name": "Wilfried Singo", "pos": "RB", "nat": "CI", "age": 25, "att": 0.16, "def": 0.26, "ovr": 80, "starter": True, "val": 18000000},
            {"name": "Kaan Ayhan", "pos": "RB", "nat": "TR", "age": 32, "att": 0.08, "def": 0.22, "ovr": 76, "starter": False, "val": 3500000},
            {"name": "Ismail Jakobs", "pos": "LB", "nat": "SN", "age": 27, "att": 0.14, "def": 0.22, "ovr": 77, "starter": True, "val": 9000000},
            {"name": "Eren Elmali", "pos": "LB", "nat": "TR", "age": 26, "att": 0.12, "def": 0.20, "ovr": 76, "starter": False, "val": 6500000},
            {"name": "Lucas Torreira", "pos": "DM", "nat": "UY", "age": 30, "att": 0.10, "def": 0.36, "ovr": 82, "starter": True, "val": 14000000},
            {"name": "Mario Lemina", "pos": "DM", "nat": "GA", "age": 33, "att": 0.10, "def": 0.28, "ovr": 77, "starter": False, "val": 4000000},
            {"name": "Berkan Kutlu", "pos": "DM", "nat": "TR", "age": 28, "att": 0.05, "def": 0.16, "ovr": 73, "starter": False, "val": 3000000},
            {"name": "Gabriel Sara", "pos": "CM", "nat": "BR", "age": 27, "att": 0.26, "def": 0.16, "ovr": 81, "starter": True, "val": 22000000},
            {"name": "Ilkay Gundogan", "pos": "CM", "nat": "DE", "age": 36, "att": 0.22, "def": 0.18, "ovr": 81, "starter": False, "val": 8000000},
            {"name": "Dries Mertens", "pos": "AM", "nat": "BE", "age": 39, "att": 0.26, "def": 0.04, "ovr": 78, "starter": True, "val": 1500000},
            {"name": "Baris Alper Yilmaz", "pos": "RW", "nat": "TR", "age": 26, "att": 0.30, "def": 0.16, "ovr": 81, "starter": True, "val": 25000000},
            {"name": "Roland Sallai", "pos": "RW", "nat": "HU", "age": 29, "att": 0.22, "def": 0.10, "ovr": 77, "starter": False, "val": 12000000},
            {"name": "Yunus Akgun", "pos": "LW", "nat": "TR", "age": 26, "att": 0.28, "def": 0.08, "ovr": 79, "starter": True, "val": 12000000},
            {"name": "Victor Osimhen", "pos": "CF", "nat": "NG", "age": 27, "att": 0.44, "def": 0.02, "ovr": 85, "starter": True, "val": 75000000},
            {"name": "Mauro Icardi", "pos": "CF", "nat": "AR", "age": 33, "att": 0.38, "def": 0.00, "ovr": 81, "starter": False, "val": 11000000},
            {"name": "Michy Batshuayi", "pos": "CF", "nat": "BE", "age": 33, "att": 0.26, "def": 0.00, "ovr": 76, "starter": False, "val": 5000000},
        ],
        "synergies": [
            ("defense", "Davinson Sanchez", "Abdulkerim Bardakci", 1.20),
            ("defense", "Ugurcan Cakir", "Davinson Sanchez", 1.18),
            ("defense", "Ugurcan Cakir", "Abdulkerim Bardakci", 1.16),
            ("midfield", "Lucas Torreira", "Gabriel Sara", 1.18),
            ("midfield", "Lucas Torreira", "Dries Mertens", 1.14),
            ("attack", "Dries Mertens", "Victor Osimhen", 1.24),
            ("attack", "Baris Alper Yilmaz", "Victor Osimhen", 1.20),
            ("attack", "Yunus Akgun", "Victor Osimhen", 1.18),
        ]
    },
    2: {  # Fenerbahce
        "name": "Fenerbahce",
        "players": [
            {"name": "Dominik Livakovic", "pos": "GK", "nat": "HR", "age": 30, "att": 0.00, "def": 0.33, "ovr": 80, "starter": True, "val": 11000000},
            {"name": "Irfan Can Egribayat", "pos": "GK", "nat": "TR", "age": 27, "att": 0.00, "def": 0.12, "ovr": 73, "starter": False, "val": 1800000},
            {"name": "Alexander Djiku", "pos": "CB", "nat": "GH", "age": 31, "att": 0.04, "def": 0.32, "ovr": 80, "starter": True, "val": 9000000},
            {"name": "Caglar Soyuncu", "pos": "CB", "nat": "TR", "age": 29, "att": 0.04, "def": 0.28, "ovr": 79, "starter": True, "val": 10000000},
            {"name": "Rodrigo Becao", "pos": "CB", "nat": "BR", "age": 29, "att": 0.03, "def": 0.26, "ovr": 78, "starter": False, "val": 8000000},
            {"name": "Mert Muldur", "pos": "RB", "nat": "TR", "age": 26, "att": 0.10, "def": 0.20, "ovr": 76, "starter": True, "val": 6000000},
            {"name": "Jayden Oosterwolde", "pos": "LB", "nat": "NL", "age": 24, "att": 0.12, "def": 0.24, "ovr": 78, "starter": True, "val": 15000000},
            {"name": "Sofyan Amrabat", "pos": "DM", "nat": "MA", "age": 29, "att": 0.10, "def": 0.32, "ovr": 80, "starter": True, "val": 22000000},
            {"name": "Ismail Yuksek", "pos": "DM", "nat": "TR", "age": 26, "att": 0.08, "def": 0.25, "ovr": 77, "starter": False, "val": 14000000},
            {"name": "Fred", "pos": "CM", "nat": "BR", "age": 32, "att": 0.22, "def": 0.32, "ovr": 82, "starter": True, "val": 15000000},
            {"name": "Mert Hakan Yandas", "pos": "CM", "nat": "TR", "age": 31, "att": 0.12, "def": 0.12, "ovr": 73, "starter": False, "val": 1500000},
            {"name": "Sebastian Szymanski", "pos": "AM", "nat": "PL", "age": 26, "att": 0.24, "def": 0.14, "ovr": 79, "starter": True, "val": 19000000},
            {"name": "Irfan Can Kahveci", "pos": "RW", "nat": "TR", "age": 30, "att": 0.26, "def": 0.12, "ovr": 79, "starter": True, "val": 10000000},
            {"name": "Oguz Aydin", "pos": "RW", "nat": "TR", "age": 25, "att": 0.18, "def": 0.08, "ovr": 74, "starter": False, "val": 4500000},
            {"name": "Dusan Tadic", "pos": "LW", "nat": "RS", "age": 36, "att": 0.32, "def": 0.08, "ovr": 81, "starter": True, "val": 3200000},
            {"name": "Allan Saint-Maximin", "pos": "LW", "nat": "FR", "age": 28, "att": 0.28, "def": 0.04, "ovr": 80, "starter": False, "val": 18000000},
            {"name": "Edin Dzeko", "pos": "CF", "nat": "BA", "age": 39, "att": 0.36, "def": 0.02, "ovr": 81, "starter": True, "val": 2500000},
            {"name": "Youssef En-Nesyri", "pos": "CF", "nat": "MA", "age": 28, "att": 0.34, "def": 0.05, "ovr": 81, "starter": False, "val": 22000000},
            {"name": "Cenk Tosun", "pos": "CF", "nat": "TR", "age": 34, "att": 0.20, "def": 0.00, "ovr": 74, "starter": False, "val": 1500000},
        ],
        "synergies": [
            ("defense", "Alexander Djiku", "Caglar Soyuncu", 1.18),
            ("defense", "Dominik Livakovic", "Alexander Djiku", 1.16),
            ("midfield", "Sofyan Amrabat", "Fred", 1.22),
            ("midfield", "Fred", "Sebastian Szymanski", 1.18),
            ("attack", "Dusan Tadic", "Edin Dzeko", 1.26),
            ("attack", "Sebastian Szymanski", "Edin Dzeko", 1.18),
            ("attack", "Irfan Can Kahveci", "Edin Dzeko", 1.16),
        ]
    },
    3: {  # Besiktas
        "name": "Besiktas",
        "players": [
            {"name": "Mert Gunok", "pos": "GK", "nat": "TR", "age": 37, "att": 0.00, "def": 0.28, "ovr": 78, "starter": True, "val": 1000000},
            {"name": "Ersin Destanoglu", "pos": "GK", "nat": "TR", "age": 25, "att": 0.00, "def": 0.14, "ovr": 74, "starter": False, "val": 2500000},
            {"name": "Gabriel Paulista", "pos": "CB", "nat": "BR", "age": 35, "att": 0.04, "def": 0.30, "ovr": 79, "starter": True, "val": 2000000},
            {"name": "Felix Uduokhai", "pos": "CB", "nat": "DE", "age": 29, "att": 0.03, "def": 0.28, "ovr": 78, "starter": True, "val": 8000000},
            {"name": "Emirhan Topcu", "pos": "CB", "nat": "TR", "age": 26, "att": 0.04, "def": 0.22, "ovr": 76, "starter": False, "val": 5000000},
            {"name": "Jonas Svensson", "pos": "RB", "nat": "NO", "age": 33, "att": 0.08, "def": 0.20, "ovr": 75, "starter": True, "val": 1500000},
            {"name": "Arthur Masuaku", "pos": "LB", "nat": "CD", "age": 32, "att": 0.14, "def": 0.16, "ovr": 76, "starter": True, "val": 3500000},
            {"name": "Al-Musrati", "pos": "DM", "nat": "LY", "age": 30, "att": 0.08, "def": 0.28, "ovr": 79, "starter": True, "val": 13000000},
            {"name": "Salih Ucan", "pos": "CM", "nat": "TR", "age": 32, "att": 0.14, "def": 0.16, "ovr": 75, "starter": False, "val": 3000000},
            {"name": "Gedson Fernandes", "pos": "CM", "nat": "PT", "age": 27, "att": 0.22, "def": 0.28, "ovr": 81, "starter": True, "val": 18000000},
            {"name": "Orkun Kokcu", "pos": "CM", "nat": "TR", "age": 25, "att": 0.24, "def": 0.18, "ovr": 82, "starter": False, "val": 25000000},
            {"name": "Rafa Silva", "pos": "AM", "nat": "PT", "age": 33, "att": 0.35, "def": 0.10, "ovr": 82, "starter": True, "val": 12000000},
            {"name": "Ernest Muci", "pos": "AM", "nat": "AL", "age": 25, "att": 0.20, "def": 0.06, "ovr": 76, "starter": False, "val": 11000000},
            {"name": "Milot Rashica", "pos": "RW", "nat": "XK", "age": 30, "att": 0.22, "def": 0.14, "ovr": 78, "starter": True, "val": 7000000},
            {"name": "Cengiz Onder", "pos": "RW", "nat": "TR", "age": 29, "att": 0.24, "def": 0.08, "ovr": 77, "starter": False, "val": 8000000},
            {"name": "Semih Kilicsoy", "pos": "LW", "nat": "TR", "age": 21, "att": 0.28, "def": 0.08, "ovr": 79, "starter": True, "val": 18000000},
            {"name": "Tammy Abraham", "pos": "CF", "nat": "GB", "age": 29, "att": 0.38, "def": 0.02, "ovr": 82, "starter": True, "val": 20000000},
            {"name": "Ciro Immobile", "pos": "CF", "nat": "IT", "age": 36, "att": 0.34, "def": 0.01, "ovr": 79, "starter": False, "val": 2800000},
            {"name": "Mustafa Hekimoglu", "pos": "CF", "nat": "TR", "age": 19, "att": 0.16, "def": 0.00, "ovr": 71, "starter": False, "val": 4000000},
        ],
        "synergies": [
            ("defense", "Gabriel Paulista", "Felix Uduokhai", 1.16),
            ("defense", "Mert Gunok", "Gabriel Paulista", 1.14),
            ("midfield", "Al-Musrati", "Gedson Fernandes", 1.18),
            ("midfield", "Gedson Fernandes", "Rafa Silva", 1.22),
            ("attack", "Rafa Silva", "Tammy Abraham", 1.25),
            ("attack", "Milot Rashica", "Tammy Abraham", 1.18),
            ("attack", "Semih Kilicsoy", "Tammy Abraham", 1.16),
        ]
    },
    4: {  # Trabzonspor
        "name": "Trabzonspor",
        "players": [
            {"name": "Andre Onana", "pos": "GK", "nat": "CM", "age": 30, "att": 0.00, "def": 0.34, "ovr": 81, "starter": True, "val": 25000000},
            {"name": "Muhammet Taha Tepe", "pos": "GK", "nat": "TR", "age": 25, "att": 0.00, "def": 0.12, "ovr": 71, "starter": False, "val": 600000},
            {"name": "Stefan Savic", "pos": "CB", "nat": "ME", "age": 35, "att": 0.04, "def": 0.30, "ovr": 78, "starter": True, "val": 2000000},
            {"name": "Serdar Saatci", "pos": "CB", "nat": "TR", "age": 23, "att": 0.03, "def": 0.26, "ovr": 76, "starter": True, "val": 4000000},
            {"name": "Stefano Denswil", "pos": "CB", "nat": "SR", "age": 33, "att": 0.04, "def": 0.24, "ovr": 75, "starter": False, "val": 1800000},
            {"name": "Pedro Malheiro", "pos": "RB", "nat": "PT", "age": 25, "att": 0.14, "def": 0.20, "ovr": 76, "starter": True, "val": 5500000},
            {"name": "Borna Barisic", "pos": "LB", "nat": "HR", "age": 33, "att": 0.14, "def": 0.18, "ovr": 75, "starter": True, "val": 2000000},
            {"name": "Batista Mendy", "pos": "DM", "nat": "FR", "age": 26, "att": 0.12, "def": 0.28, "ovr": 79, "starter": True, "val": 12000000},
            {"name": "Okay Yokuslu", "pos": "DM", "nat": "TR", "age": 32, "att": 0.08, "def": 0.26, "ovr": 77, "starter": True, "val": 3000000},
            {"name": "John Lundstram", "pos": "CM", "nat": "GB", "age": 32, "att": 0.12, "def": 0.20, "ovr": 74, "starter": False, "val": 2500000},
            {"name": "Muhammed Cham", "pos": "AM", "nat": "AT", "age": 25, "att": 0.24, "def": 0.06, "ovr": 77, "starter": True, "val": 6000000},
            {"name": "Edin Visca", "pos": "RW", "nat": "BA", "age": 36, "att": 0.28, "def": 0.08, "ovr": 77, "starter": True, "val": 1800000},
            {"name": "Anthony Nwakaeme", "pos": "LW", "nat": "NG", "age": 37, "att": 0.25, "def": 0.04, "ovr": 76, "starter": True, "val": 1000000},
            {"name": "Cihan Canak", "pos": "LW", "nat": "TR", "age": 21, "att": 0.18, "def": 0.06, "ovr": 73, "starter": False, "val": 3500000},
            {"name": "Denis Dragus", "pos": "LW", "nat": "RO", "age": 27, "att": 0.22, "def": 0.06, "ovr": 75, "starter": False, "val": 4000000},
            {"name": "Simon Banza", "pos": "CF", "nat": "CD", "age": 30, "att": 0.38, "def": 0.02, "ovr": 82, "starter": True, "val": 16000000},
            {"name": "Enis Destan", "pos": "CF", "nat": "TR", "age": 24, "att": 0.18, "def": 0.01, "ovr": 73, "starter": False, "val": 4000000},
        ],
        "synergies": [
            ("defense", "Andre Onana", "Stefan Savic", 1.15),
            ("defense", "Stefan Savic", "Serdar Saatci", 1.14),
            ("midfield", "Batista Mendy", "Okay Yokuslu", 1.18),
            ("attack", "Edin Visca", "Simon Banza", 1.22),
            ("attack", "Muhammed Cham", "Simon Banza", 1.18),
        ]
    },
    5: {  # Basaksehir
        "name": "Basaksehir",
        "players": [
            {"name": "Muhammed Sengezer", "pos": "GK", "nat": "TR", "age": 29, "att": 0.00, "def": 0.28, "ovr": 77, "starter": True, "val": 3000000},
            {"name": "Jerome Opoku", "pos": "CB", "nat": "GH", "age": 27, "att": 0.06, "def": 0.28, "ovr": 78, "starter": True, "val": 6000000},
            {"name": "Leo Duarte", "pos": "CB", "nat": "BR", "age": 30, "att": 0.03, "def": 0.26, "ovr": 76, "starter": True, "val": 3000000},
            {"name": "Festy Ebosele", "pos": "RB", "nat": "IE", "age": 24, "att": 0.16, "def": 0.18, "ovr": 76, "starter": True, "val": 5000000},
            {"name": "Christopher Operi", "pos": "LB", "nat": "CI", "age": 29, "att": 0.14, "def": 0.18, "ovr": 75, "starter": True, "val": 3000000},
            {"name": "Berat Ozdemir", "pos": "DM", "nat": "TR", "age": 28, "att": 0.08, "def": 0.24, "ovr": 76, "starter": True, "val": 4000000},
            {"name": "Olivier Kemen", "pos": "CM", "nat": "CM", "age": 30, "att": 0.14, "def": 0.20, "ovr": 76, "starter": True, "val": 3500000},
            {"name": "Miguel Crespo", "pos": "CM", "nat": "PT", "age": 30, "att": 0.12, "def": 0.22, "ovr": 76, "starter": False, "val": 4000000},
            {"name": "Deniz Turuc", "pos": "AM", "nat": "TR", "age": 33, "att": 0.22, "def": 0.12, "ovr": 76, "starter": True, "val": 2000000},
            {"name": "Amine Harit", "pos": "RW", "nat": "MA", "age": 29, "att": 0.26, "def": 0.08, "ovr": 78, "starter": True, "val": 9000000},
            {"name": "Abbosbek Fayzullaev", "pos": "LW", "nat": "UZ", "age": 23, "att": 0.28, "def": 0.08, "ovr": 79, "starter": True, "val": 12000000},
            {"name": "Eldor Shomurodov", "pos": "CF", "nat": "UZ", "age": 31, "att": 0.32, "def": 0.02, "ovr": 78, "starter": True, "val": 6500000},
            {"name": "Bertug Yildirim", "pos": "CF", "nat": "TR", "age": 24, "att": 0.26, "def": 0.02, "ovr": 76, "starter": False, "val": 5000000},
            {"name": "Krzysztof Piatek", "pos": "CF", "nat": "PL", "age": 31, "att": 0.32, "def": 0.01, "ovr": 79, "starter": False, "val": 5500000},
        ],
        "synergies": [
            ("defense", "Leo Duarte", "Jerome Opoku", 1.16),
            ("midfield", "Berat Ozdemir", "Olivier Kemen", 1.16),
            ("attack", "Amine Harit", "Eldor Shomurodov", 1.20),
            ("attack", "Abbosbek Fayzullaev", "Eldor Shomurodov", 1.20),
        ]
    },
    5: {  # Basaksehir
        "name": "Basaksehir",
        "players": [
            {"name": "Muhammed Sengezer", "pos": "GK", "nat": "TR", "age": 28, "att": 0.00, "def": 0.24, "ovr": 76, "starter": True, "val": 2500000},
            {"name": "Volkan Babacan", "pos": "GK", "nat": "TR", "age": 37, "att": 0.00, "def": 0.10, "ovr": 70, "starter": False, "val": 200000},
            {"name": "Leo Duarte", "pos": "CB", "nat": "BR", "age": 29, "att": 0.03, "def": 0.24, "ovr": 76, "starter": True, "val": 3000000},
            {"name": "Jerome Opoku", "pos": "CB", "nat": "GH", "age": 27, "att": 0.05, "def": 0.22, "ovr": 75, "starter": True, "val": 3500000},
            {"name": "Olivier Kemen", "pos": "RB", "nat": "CM", "age": 29, "att": 0.10, "def": 0.18, "ovr": 75, "starter": True, "val": 2800000},
            {"name": "Lucas Lima", "pos": "LB", "nat": "BR", "age": 34, "att": 0.10, "def": 0.18, "ovr": 74, "starter": True, "val": 1200000},
            {"name": "Berat Ozdemir", "pos": "DM", "nat": "TR", "age": 27, "att": 0.08, "def": 0.22, "ovr": 75, "starter": True, "val": 4500000},
            {"name": "Miguel Crespo", "pos": "CM", "nat": "PT", "age": 29, "att": 0.14, "def": 0.22, "ovr": 77, "starter": True, "val": 4000000},
            {"name": "Onur Ergun", "pos": "CM", "nat": "TR", "age": 33, "att": 0.08, "def": 0.14, "ovr": 72, "starter": False, "val": 600000},
            {"name": "Dimitris Pelkas", "pos": "AM", "nat": "GR", "age": 32, "att": 0.22, "def": 0.08, "ovr": 76, "starter": True, "val": 2000000},
            {"name": "Deniz Turuc", "pos": "RW", "nat": "TR", "age": 32, "att": 0.24, "def": 0.12, "ovr": 77, "starter": True, "val": 2200000},
            {"name": "Serdar Gurler", "pos": "LW", "nat": "TR", "age": 34, "att": 0.20, "def": 0.08, "ovr": 75, "starter": True, "val": 1200000},
            {"name": "Davidson", "pos": "LW", "nat": "BR", "age": 34, "att": 0.18, "def": 0.06, "ovr": 74, "starter": False, "val": 1500000},
            {"name": "Krzysztof Piatek", "pos": "CF", "nat": "PL", "age": 30, "att": 0.32, "def": 0.01, "ovr": 79, "starter": True, "val": 6000000},
            {"name": "Joao Figueiredo", "pos": "CF", "nat": "BR", "age": 29, "att": 0.20, "def": 0.02, "ovr": 74, "starter": False, "val": 2400000},
        ],
        "synergies": [
            ("defense", "Leo Duarte", "Jerome Opoku", 1.15),
            ("midfield", "Berat Ozdemir", "Miguel Crespo", 1.16),
            ("attack", "Deniz Turuc", "Krzysztof Piatek", 1.20),
            ("attack", "Dimitris Pelkas", "Krzysztof Piatek", 1.18),
        ]
    },
    20: {  # Samsunspor
        "name": "Samsunspor",
        "players": [
            {"name": "Okan Kocuk", "pos": "GK", "nat": "TR", "age": 31, "att": 0.00, "def": 0.26, "ovr": 76, "starter": True, "val": 2200000},
            {"name": "Strahinja Erakovic", "pos": "CB", "nat": "RS", "age": 25, "att": 0.04, "def": 0.28, "ovr": 78, "starter": True, "val": 7500000},
            {"name": "Rick van Drongelen", "pos": "CB", "nat": "NL", "age": 27, "att": 0.04, "def": 0.26, "ovr": 76, "starter": True, "val": 2500000},
            {"name": "Joe Mendes", "pos": "RB", "nat": "SE", "age": 23, "att": 0.14, "def": 0.18, "ovr": 74, "starter": True, "val": 2000000},
            {"name": "Logi Tomasson", "pos": "LB", "nat": "IS", "age": 25, "att": 0.14, "def": 0.18, "ovr": 74, "starter": True, "val": 2200000},
            {"name": "Antoine Sekongo", "pos": "DM", "nat": "CI", "age": 22, "att": 0.08, "def": 0.22, "ovr": 74, "starter": True, "val": 2500000},
            {"name": "Celil Yuksel", "pos": "CM", "nat": "TR", "age": 28, "att": 0.14, "def": 0.18, "ovr": 74, "starter": True, "val": 1800000},
            {"name": "Afonso Sousa", "pos": "AM", "nat": "PT", "age": 26, "att": 0.24, "def": 0.10, "ovr": 76, "starter": True, "val": 4000000},
            {"name": "Emre Kilinc", "pos": "RW", "nat": "TR", "age": 32, "att": 0.22, "def": 0.12, "ovr": 76, "starter": True, "val": 2200000},
            {"name": "Tanguy Coulibaly", "pos": "LW", "nat": "FR", "age": 25, "att": 0.24, "def": 0.08, "ovr": 76, "starter": True, "val": 3500000},
            {"name": "Mohamed Bayo", "pos": "CF", "nat": "GN", "age": 28, "att": 0.32, "def": 0.02, "ovr": 78, "starter": True, "val": 6500000},
            {"name": "Marius Mouandilmadji", "pos": "CF", "nat": "TD", "age": 28, "att": 0.26, "def": 0.02, "ovr": 76, "starter": False, "val": 2500000},
        ],
        "synergies": [
            ("defense", "Strahinja Erakovic", "Rick van Drongelen", 1.16),
            ("midfield", "Antoine Sekongo", "Afonso Sousa", 1.18),
            ("attack", "Afonso Sousa", "Mohamed Bayo", 1.20),
            ("attack", "Emre Kilinc", "Mohamed Bayo", 1.18),
        ]
    },
    17: {  # Goztepe
        "name": "Goztepe",
        "players": [
            {"name": "Luka Gugeshashvili", "pos": "GK", "nat": "GE", "age": 27, "att": 0.00, "def": 0.26, "ovr": 76, "starter": True, "val": 2500000},
            {"name": "Taha Altikardes", "pos": "CB", "nat": "TR", "age": 23, "att": 0.05, "def": 0.26, "ovr": 76, "starter": True, "val": 4500000},
            {"name": "Malcom Bokele", "pos": "CB", "nat": "CM", "age": 26, "att": 0.04, "def": 0.26, "ovr": 76, "starter": True, "val": 3500000},
            {"name": "Ogun Bayrak", "pos": "RB", "nat": "TR", "age": 28, "att": 0.12, "def": 0.18, "ovr": 74, "starter": True, "val": 1800000},
            {"name": "Richard Akonnor", "pos": "LB", "nat": "GH", "age": 23, "att": 0.12, "def": 0.16, "ovr": 73, "starter": True, "val": 1500000},
            {"name": "Novatus Miroshi", "pos": "DM", "nat": "TZ", "age": 24, "att": 0.08, "def": 0.22, "ovr": 74, "starter": True, "val": 2500000},
            {"name": "Tino Anjorin", "pos": "CM", "nat": "GB", "age": 24, "att": 0.18, "def": 0.16, "ovr": 76, "starter": True, "val": 4000000},
            {"name": "Efkan Bekiroglu", "pos": "AM", "nat": "TR", "age": 31, "att": 0.24, "def": 0.10, "ovr": 76, "starter": True, "val": 2500000},
            {"name": "Janderson", "pos": "RW", "nat": "BR", "age": 27, "att": 0.22, "def": 0.08, "ovr": 74, "starter": True, "val": 2000000},
            {"name": "Romulo", "pos": "CF", "nat": "BR", "age": 24, "att": 0.38, "def": 0.02, "ovr": 81, "starter": True, "val": 9000000},
            {"name": "Juan", "pos": "CF", "nat": "BR", "age": 24, "att": 0.28, "def": 0.02, "ovr": 76, "starter": False, "val": 3500000},
            {"name": "Furkan Bayir", "pos": "CB", "nat": "TR", "age": 26, "att": 0.03, "def": 0.22, "ovr": 74, "starter": False, "val": 2000000},
        ],
        "synergies": [
            ("defense", "Taha Altikardes", "Malcom Bokele", 1.16),
            ("midfield", "Novatus Miroshi", "Tino Anjorin", 1.18),
            ("attack", "Efkan Bekiroglu", "Romulo", 1.22),
            ("attack", "Janderson", "Romulo", 1.18),
        ]
    },
    26: {  # Eyupspor
        "name": "Eyupspor",
        "players": [
            {"name": "Berke Ozer", "pos": "GK", "nat": "TR", "age": 25, "att": 0.00, "def": 0.26, "ovr": 76, "starter": True, "val": 3000000},
            {"name": "Luccas Claro", "pos": "CB", "nat": "BR", "age": 34, "att": 0.04, "def": 0.24, "ovr": 75, "starter": True, "val": 1000000},
            {"name": "Ruben Vezo", "pos": "CB", "nat": "PT", "age": 31, "att": 0.02, "def": 0.22, "ovr": 75, "starter": True, "val": 2000000},
            {"name": "Leo Dubois", "pos": "RB", "nat": "FR", "age": 31, "att": 0.10, "def": 0.20, "ovr": 75, "starter": True, "val": 2500000},
            {"name": "Caner Erkin", "pos": "LB", "nat": "TR", "age": 37, "att": 0.16, "def": 0.12, "ovr": 73, "starter": True, "val": 300000},
            {"name": "Robin Yalcin", "pos": "DM", "nat": "TR", "age": 31, "att": 0.08, "def": 0.20, "ovr": 74, "starter": True, "val": 1200000},
            {"name": "Fredrik Midtsjo", "pos": "CM", "nat": "NO", "age": 32, "att": 0.14, "def": 0.22, "ovr": 76, "starter": True, "val": 2500000},
            {"name": "Emre Akbaba", "pos": "AM", "nat": "TR", "age": 33, "att": 0.24, "def": 0.08, "ovr": 76, "starter": True, "val": 1800000},
            {"name": "Samu Saiz", "pos": "RW", "nat": "ES", "age": 34, "att": 0.22, "def": 0.06, "ovr": 75, "starter": True, "val": 1200000},
            {"name": "Ahmed Kutucu", "pos": "LW", "nat": "TR", "age": 25, "att": 0.25, "def": 0.08, "ovr": 77, "starter": True, "val": 4000000},
            {"name": "Mame Thiam", "pos": "CF", "nat": "SN", "age": 33, "att": 0.28, "def": 0.02, "ovr": 77, "starter": True, "val": 1800000},
            {"name": "Gianni Bruno", "pos": "CF", "nat": "BE", "age": 34, "att": 0.16, "def": 0.00, "ovr": 72, "starter": False, "val": 600000},
        ],
        "synergies": [
            ("defense", "Luccas Claro", "Ruben Vezo", 1.15),
            ("midfield", "Fredrik Midtsjo", "Emre Akbaba", 1.18),
            ("attack", "Ahmed Kutucu", "Mame Thiam", 1.20),
            ("attack", "Emre Akbaba", "Mame Thiam", 1.20),
        ]
    },
    39: {  # Amedspor
        "name": "Amedspor",
        "players": [
            {"name": "Nurullah Aslan", "pos": "GK", "nat": "TR", "age": 28, "att": 0.00, "def": 0.20, "ovr": 74, "starter": True, "val": 800000},
            {"name": "Mehmet Yesil", "pos": "CB", "nat": "TR", "age": 27, "att": 0.02, "def": 0.22, "ovr": 74, "starter": True, "val": 900000},
            {"name": "Ugur Adem Gezer", "pos": "CB", "nat": "TR", "age": 31, "att": 0.02, "def": 0.20, "ovr": 73, "starter": True, "val": 600000},
            {"name": "Murat Ucar", "pos": "RB", "nat": "TR", "age": 34, "att": 0.08, "def": 0.16, "ovr": 72, "starter": True, "val": 300000},
            {"name": "Alberk Koc", "pos": "LB", "nat": "TR", "age": 28, "att": 0.08, "def": 0.16, "ovr": 72, "starter": True, "val": 500000},
            {"name": "Yohan Cassubie", "pos": "DM", "nat": "FR", "age": 25, "att": 0.08, "def": 0.22, "ovr": 75, "starter": True, "val": 1500000},
            {"name": "Adama Traore", "pos": "CM", "nat": "ML", "age": 30, "att": 0.20, "def": 0.16, "ovr": 77, "starter": True, "val": 2200000},
            {"name": "Cmiljanic", "pos": "AM", "nat": "ME", "age": 29, "att": 0.18, "def": 0.08, "ovr": 73, "starter": True, "val": 800000},
            {"name": "Max Gradel", "pos": "LW", "nat": "CI", "age": 38, "att": 0.22, "def": 0.06, "ovr": 75, "starter": True, "val": 600000},
            {"name": "Oktay Aydin", "pos": "RW", "nat": "TR", "age": 28, "att": 0.16, "def": 0.10, "ovr": 73, "starter": True, "val": 600000},
            {"name": "Britt Assombalonga", "pos": "CF", "nat": "CD", "age": 33, "att": 0.26, "def": 0.01, "ovr": 76, "starter": True, "val": 1200000},
            {"name": "Yakubu", "pos": "CF", "nat": "GH", "age": 24, "att": 0.16, "def": 0.00, "ovr": 71, "starter": False, "val": 600000},
        ],
        "synergies": [
            ("defense", "Mehmet Yesil", "Ugur Adem Gezer", 1.15),
            ("midfield", "Yohan Cassubie", "Adama Traore", 1.18),
            ("attack", "Max Gradel", "Britt Assombalonga", 1.20),
        ]
    },
    38: {  # Corum FK
        "name": "Corum FK",
        "players": [
            {"name": "Hasan Huseyin Akinay", "pos": "GK", "nat": "TR", "age": 32, "att": 0.00, "def": 0.18, "ovr": 73, "starter": True, "val": 600000},
            {"name": "Zargo Toure", "pos": "CB", "nat": "SN", "age": 36, "att": 0.02, "def": 0.22, "ovr": 74, "starter": True, "val": 400000},
            {"name": "Loick Landre", "pos": "CB", "nat": "FR", "age": 33, "att": 0.02, "def": 0.20, "ovr": 73, "starter": True, "val": 450000},
            {"name": "Kerem Kalafat", "pos": "RB", "nat": "TR", "age": 24, "att": 0.08, "def": 0.16, "ovr": 72, "starter": True, "val": 500000},
            {"name": "Erkan Kas", "pos": "LB", "nat": "TR", "age": 34, "att": 0.08, "def": 0.16, "ovr": 71, "starter": True, "val": 250000},
            {"name": "Ferhat Yazgan", "pos": "DM", "nat": "TR", "age": 33, "att": 0.08, "def": 0.20, "ovr": 73, "starter": True, "val": 500000},
            {"name": "Atakan Akkaynak", "pos": "CM", "nat": "DE", "age": 27, "att": 0.14, "def": 0.16, "ovr": 74, "starter": True, "val": 800000},
            {"name": "Mehdi Boudjemaa", "pos": "CM", "nat": "DZ", "age": 27, "att": 0.12, "def": 0.18, "ovr": 73, "starter": True, "val": 800000},
            {"name": "Geraldo", "pos": "RW", "nat": "AO", "age": 34, "att": 0.20, "def": 0.08, "ovr": 74, "starter": True, "val": 500000},
            {"name": "Ozan Sol", "pos": "LW", "nat": "TR", "age": 32, "att": 0.18, "def": 0.06, "ovr": 72, "starter": True, "val": 400000},
            {"name": "Thomas Verheydt", "pos": "CF", "nat": "NL", "age": 33, "att": 0.25, "def": 0.01, "ovr": 75, "starter": True, "val": 800000},
            {"name": "Ahmethan Kose", "pos": "CF", "nat": "TR", "age": 28, "att": 0.14, "def": 0.00, "ovr": 70, "starter": False, "val": 350000},
        ],
        "synergies": [
            ("defense", "Zargo Toure", "Loick Landre", 1.15),
            ("midfield", "Ferhat Yazgan", "Atakan Akkaynak", 1.16),
            ("attack", "Geraldo", "Thomas Verheydt", 1.18),
        ]
    },
    6: {  # Konyaspor
        "name": "Konyaspor",
        "players": [
            {"name": "Jakub Slowik", "pos": "GK", "nat": "PL", "age": 34, "att": 0.00, "def": 0.24, "ovr": 75, "starter": True, "val": 800000},
            {"name": "Adil Demirbag", "pos": "CB", "nat": "TR", "age": 28, "att": 0.02, "def": 0.22, "ovr": 75, "starter": True, "val": 2200000},
            {"name": "Riechedly Bazoer", "pos": "CB", "nat": "NL", "age": 29, "att": 0.06, "def": 0.24, "ovr": 76, "starter": True, "val": 2800000},
            {"name": "Nikola Boranijasevic", "pos": "RB", "nat": "RS", "age": 33, "att": 0.08, "def": 0.16, "ovr": 73, "starter": True, "val": 600000},
            {"name": "Guilherme Sitya", "pos": "LB", "nat": "BR", "age": 35, "att": 0.16, "def": 0.18, "ovr": 76, "starter": True, "val": 1500000},
            {"name": "Ugurcan Yazgili", "pos": "DM", "nat": "TR", "age": 26, "att": 0.06, "def": 0.20, "ovr": 74, "starter": True, "val": 1800000},
            {"name": "Marko Jevtovic", "pos": "CM", "nat": "RS", "age": 32, "att": 0.12, "def": 0.20, "ovr": 74, "starter": True, "val": 1000000},
            {"name": "Danijel Aleksic", "pos": "AM", "nat": "RS", "age": 34, "att": 0.20, "def": 0.08, "ovr": 75, "starter": True, "val": 1000000},
            {"name": "Pedrinho", "pos": "RW", "nat": "PT", "age": 33, "att": 0.22, "def": 0.08, "ovr": 75, "starter": True, "val": 1200000},
            {"name": "Yusuf Erdogan", "pos": "LW", "nat": "TR", "age": 33, "att": 0.18, "def": 0.10, "ovr": 74, "starter": True, "val": 1000000},
            {"name": "Blaz Kramer", "pos": "CF", "nat": "SI", "age": 29, "att": 0.24, "def": 0.01, "ovr": 75, "starter": True, "val": 1800000},
            {"name": "Umut Nayir", "pos": "CF", "nat": "TR", "age": 32, "att": 0.20, "def": 0.01, "ovr": 74, "starter": False, "val": 1200000},
        ],
        "synergies": [
            ("defense", "Adil Demirbag", "Riechedly Bazoer", 1.15),
            ("midfield", "Ugurcan Yazgili", "Marko Jevtovic", 1.15),
            ("attack", "Pedrinho", "Blaz Kramer", 1.18),
        ]
    },
    8: {  # Kasimpasa
        "name": "Kasimpasa",
        "players": [
            {"name": "Andreas Gianniotis", "pos": "GK", "nat": "GR", "age": 33, "att": 0.00, "def": 0.26, "ovr": 76, "starter": True, "val": 1500000},
            {"name": "Attila Szalai", "pos": "CB", "nat": "HU", "age": 28, "att": 0.04, "def": 0.28, "ovr": 77, "starter": True, "val": 4500000},
            {"name": "Kenneth Omeruo", "pos": "CB", "nat": "NG", "age": 32, "att": 0.04, "def": 0.24, "ovr": 76, "starter": True, "val": 1800000},
            {"name": "Claudio Winck", "pos": "RB", "nat": "BR", "age": 32, "att": 0.14, "def": 0.18, "ovr": 75, "starter": True, "val": 2000000},
            {"name": "Kevin Rodrigues", "pos": "LB", "nat": "PT", "age": 32, "att": 0.14, "def": 0.16, "ovr": 74, "starter": True, "val": 1800000},
            {"name": "Loret Sadiku", "pos": "DM", "nat": "XK", "age": 35, "att": 0.06, "def": 0.22, "ovr": 73, "starter": True, "val": 800000},
            {"name": "Aytac Kara", "pos": "CM", "nat": "TR", "age": 33, "att": 0.22, "def": 0.16, "ovr": 76, "starter": True, "val": 2200000},
            {"name": "Haris Hajradinovic", "pos": "AM", "nat": "BA", "age": 32, "att": 0.24, "def": 0.10, "ovr": 77, "starter": True, "val": 3000000},
            {"name": "Mamadou Fall", "pos": "RW", "nat": "SN", "age": 34, "att": 0.22, "def": 0.10, "ovr": 75, "starter": True, "val": 1500000},
            {"name": "Mortadha Ben Ouanes", "pos": "LW", "nat": "TN", "age": 32, "att": 0.22, "def": 0.14, "ovr": 75, "starter": True, "val": 2000000},
            {"name": "Nuno Da Costa", "pos": "CF", "nat": "CV", "age": 35, "att": 0.32, "def": 0.02, "ovr": 77, "starter": True, "val": 1800000},
            {"name": "Josip Brekalo", "pos": "LW", "nat": "HR", "age": 28, "att": 0.22, "def": 0.06, "ovr": 76, "starter": False, "val": 3500000},
        ],
        "synergies": [
            ("defense", "Attila Szalai", "Kenneth Omeruo", 1.16),
            ("midfield", "Aytac Kara", "Haris Hajradinovic", 1.20),
            ("attack", "Haris Hajradinovic", "Nuno Da Costa", 1.24),
            ("attack", "Mamadou Fall", "Nuno Da Costa", 1.18),
        ]
    },
    10: {  # Alanyaspor
        "name": "Alanyaspor",
        "players": [
            {"name": "Ertugrul Taskiran", "pos": "GK", "nat": "TR", "age": 36, "att": 0.00, "def": 0.22, "ovr": 74, "starter": True, "val": 500000},
            {"name": "Furkan Bayir", "pos": "CB", "nat": "TR", "age": 25, "att": 0.02, "def": 0.22, "ovr": 74, "starter": True, "val": 2000000},
            {"name": "Fidan Aliti", "pos": "CB", "nat": "XK", "age": 32, "att": 0.02, "def": 0.22, "ovr": 74, "starter": True, "val": 1400000},
            {"name": "Florent Hadergjonaj", "pos": "RB", "nat": "XK", "age": 31, "att": 0.12, "def": 0.18, "ovr": 75, "starter": True, "val": 1800000},
            {"name": "Jure Balkovec", "pos": "LB", "nat": "SI", "age": 31, "att": 0.12, "def": 0.18, "ovr": 74, "starter": True, "val": 1200000},
            {"name": "Richard", "pos": "DM", "nat": "BR", "age": 31, "att": 0.10, "def": 0.24, "ovr": 76, "starter": True, "val": 2200000},
            {"name": "Gaius Makouta", "pos": "CM", "nat": "CG", "age": 28, "att": 0.16, "def": 0.18, "ovr": 75, "starter": True, "val": 2500000},
            {"name": "Efecan Karaca", "pos": "RW", "nat": "TR", "age": 36, "att": 0.20, "def": 0.12, "ovr": 74, "starter": True, "val": 600000},
            {"name": "Nicolas Janvier", "pos": "AM", "nat": "FR", "age": 27, "att": 0.18, "def": 0.10, "ovr": 74, "starter": True, "val": 1600000},
            {"name": "Loide Augusto", "pos": "LW", "nat": "AO", "age": 25, "att": 0.20, "def": 0.06, "ovr": 74, "starter": True, "val": 2000000},
            {"name": "Sergio Cordova", "pos": "CF", "nat": "VE", "age": 28, "att": 0.24, "def": 0.01, "ovr": 75, "starter": True, "val": 2500000},
            {"name": "Hwang Ui-jo", "pos": "CF", "nat": "KR", "age": 33, "att": 0.22, "def": 0.01, "ovr": 75, "starter": False, "val": 1800000},
        ],
        "synergies": [
            ("defense", "Furkan Bayir", "Fidan Aliti", 1.16),
            ("midfield", "Richard", "Gaius Makouta", 1.18),
            ("attack", "Efecan Karaca", "Sergio Cordova", 1.18),
        ]
    },
    12: {  # Gaziantep FK
        "name": "Gaziantep FK",
        "players": [
            {"name": "Mustafa Burak Bozan", "pos": "GK", "nat": "TR", "age": 25, "att": 0.00, "def": 0.20, "ovr": 73, "starter": True, "val": 800000},
            {"name": "Arda Kizildag", "pos": "CB", "nat": "TR", "age": 27, "att": 0.03, "def": 0.22, "ovr": 74, "starter": True, "val": 1800000},
            {"name": "Enric Saborit", "pos": "CB", "nat": "ES", "age": 33, "att": 0.03, "def": 0.20, "ovr": 74, "starter": True, "val": 800000},
            {"name": "Salem M'Bakata", "pos": "RB", "nat": "CD", "age": 27, "att": 0.10, "def": 0.18, "ovr": 74, "starter": True, "val": 1500000},
            {"name": "Godfrey Bitok", "pos": "LB", "nat": "NG", "age": 26, "att": 0.08, "def": 0.16, "ovr": 73, "starter": True, "val": 1000000},
            {"name": "Badou Ndiaye", "pos": "DM", "nat": "SN", "age": 35, "att": 0.10, "def": 0.22, "ovr": 75, "starter": True, "val": 800000},
            {"name": "Kacper Kozlowski", "pos": "CM", "nat": "PL", "age": 22, "att": 0.20, "def": 0.14, "ovr": 76, "starter": True, "val": 5000000},
            {"name": "Alexandru Maxim", "pos": "AM", "nat": "RO", "age": 35, "att": 0.25, "def": 0.06, "ovr": 76, "starter": True, "val": 1200000},
            {"name": "Mustafa Eskihellac", "pos": "RW", "nat": "TR", "age": 28, "att": 0.18, "def": 0.14, "ovr": 74, "starter": True, "val": 2000000},
            {"name": "Deian Sorescu", "pos": "LW", "nat": "RO", "age": 28, "att": 0.20, "def": 0.12, "ovr": 75, "starter": True, "val": 2500000},
            {"name": "David Okereke", "pos": "CF", "nat": "NG", "age": 28, "att": 0.26, "def": 0.01, "ovr": 76, "starter": True, "val": 3000000},
            {"name": "Halil Dervisoglu", "pos": "CF", "nat": "TR", "age": 26, "att": 0.18, "def": 0.01, "ovr": 73, "starter": False, "val": 1800000},
        ],
        "synergies": [
            ("defense", "Arda Kizildag", "Enric Saborit", 1.15),
            ("midfield", "Badou Ndiaye", "Kacper Kozlowski", 1.16),
            ("attack", "Alexandru Maxim", "David Okereke", 1.22),
        ]
    },
    15: {  # Rizespor
        "name": "Rizespor",
        "players": [
            {"name": "Yahia Fofana", "pos": "GK", "nat": "CI", "age": 26, "att": 0.00, "def": 0.28, "ovr": 77, "starter": True, "val": 4000000},
            {"name": "Attila Mocsi", "pos": "CB", "nat": "HU", "age": 26, "att": 0.04, "def": 0.26, "ovr": 76, "starter": True, "val": 3500000},
            {"name": "Modibo Sagnan", "pos": "CB", "nat": "ML", "age": 27, "att": 0.04, "def": 0.26, "ovr": 76, "starter": True, "val": 4000000},
            {"name": "Muhammet Taha Sahin", "pos": "RB", "nat": "TR", "age": 25, "att": 0.10, "def": 0.18, "ovr": 74, "starter": True, "val": 2200000},
            {"name": "Casper Hojer", "pos": "LB", "nat": "DK", "age": 31, "att": 0.14, "def": 0.18, "ovr": 75, "starter": True, "val": 1500000},
            {"name": "Qazim Laci", "pos": "DM", "nat": "AL", "age": 30, "att": 0.10, "def": 0.24, "ovr": 76, "starter": True, "val": 3000000},
            {"name": "Ibrahim Olawoyin", "pos": "CM", "nat": "NG", "age": 28, "att": 0.20, "def": 0.18, "ovr": 77, "starter": True, "val": 4500000},
            {"name": "Can Bozdogan", "pos": "AM", "nat": "DE", "age": 25, "att": 0.22, "def": 0.12, "ovr": 76, "starter": True, "val": 3800000},
            {"name": "David Akintola", "pos": "RW", "nat": "NG", "age": 30, "att": 0.22, "def": 0.10, "ovr": 75, "starter": True, "val": 2000000},
            {"name": "Emrecan Bulut", "pos": "LW", "nat": "TR", "age": 23, "att": 0.22, "def": 0.08, "ovr": 74, "starter": True, "val": 2200000},
            {"name": "Ali Sowe", "pos": "CF", "nat": "GM", "age": 32, "att": 0.34, "def": 0.02, "ovr": 78, "starter": True, "val": 3500000},
            {"name": "Gennaro Borrelli", "pos": "CF", "nat": "IT", "age": 26, "att": 0.26, "def": 0.02, "ovr": 75, "starter": False, "val": 2500000},
        ],
        "synergies": [
            ("defense", "Attila Mocsi", "Modibo Sagnan", 1.16),
            ("midfield", "Qazim Laci", "Ibrahim Olawoyin", 1.18),
            ("attack", "Can Bozdogan", "Ali Sowe", 1.20),
            ("attack", "David Akintola", "Ali Sowe", 1.18),
        ]
    },
    27: {  # Genclerbirligi
        "name": "Genclerbirligi",
        "players": [
            {"name": "Ertugrul Cetin", "pos": "GK", "nat": "TR", "age": 22, "att": 0.00, "def": 0.18, "ovr": 72, "starter": True, "val": 500000},
            {"name": "Alperen Babacan", "pos": "CB", "nat": "TR", "age": 28, "att": 0.02, "def": 0.20, "ovr": 73, "starter": True, "val": 600000},
            {"name": "Sinan Osmanoglu", "pos": "CB", "nat": "TR", "age": 35, "att": 0.02, "def": 0.18, "ovr": 72, "starter": True, "val": 300000},
            {"name": "Yasin Gureler", "pos": "LB", "nat": "TR", "age": 34, "att": 0.08, "def": 0.16, "ovr": 71, "starter": True, "val": 250000},
            {"name": "Oguzhan Berber", "pos": "RB", "nat": "TR", "age": 33, "att": 0.08, "def": 0.16, "ovr": 71, "starter": True, "val": 250000},
            {"name": "Mikail Okyar", "pos": "DM", "nat": "TR", "age": 26, "att": 0.06, "def": 0.18, "ovr": 72, "starter": True, "val": 500000},
            {"name": "Ensar Kemaloglu", "pos": "CM", "nat": "TR", "age": 27, "att": 0.12, "def": 0.14, "ovr": 73, "starter": True, "val": 700000},
            {"name": "Amilton", "pos": "RW", "nat": "BR", "age": 36, "att": 0.20, "def": 0.06, "ovr": 74, "starter": True, "val": 400000},
            {"name": "Mete Kaan Demir", "pos": "LW", "nat": "TR", "age": 27, "att": 0.18, "def": 0.06, "ovr": 73, "starter": True, "val": 600000},
            {"name": "Bughra Cagiran", "pos": "AM", "nat": "TR", "age": 30, "att": 0.16, "def": 0.10, "ovr": 72, "starter": True, "val": 450000},
            {"name": "Mustapha Yatabare", "pos": "CF", "nat": "ML", "age": 39, "att": 0.22, "def": 0.01, "ovr": 73, "starter": True, "val": 300000},
            {"name": "Chukwuma Akabueze", "pos": "LW", "nat": "NG", "age": 36, "att": 0.16, "def": 0.04, "ovr": 71, "starter": False, "val": 250000},
        ],
        "synergies": [
            ("defense", "Alperen Babacan", "Sinan Osmanoglu", 1.14),
            ("midfield", "Mikail Okyar", "Ensar Kemaloglu", 1.15),
            ("attack", "Amilton", "Mustapha Yatabare", 1.16),
        ]
    },
    30: {  # Erzurumspor FK
        "name": "Erzurumspor FK",
        "players": [
            {"name": "Mehmet Goktug Bakirbas", "pos": "GK", "nat": "TR", "age": 29, "att": 0.00, "def": 0.20, "ovr": 73, "starter": True, "val": 700000},
            {"name": "Mustafa Yumlu", "pos": "CB", "nat": "TR", "age": 38, "att": 0.02, "def": 0.22, "ovr": 73, "starter": True, "val": 250000},
            {"name": "Orhan Ovacikli", "pos": "RB", "nat": "TR", "age": 37, "att": 0.06, "def": 0.16, "ovr": 71, "starter": True, "val": 200000},
            {"name": "Yakup Kisa", "pos": "CB", "nat": "TR", "age": 26, "att": 0.02, "def": 0.18, "ovr": 72, "starter": True, "val": 450000},
            {"name": "Gurhan Colak", "pos": "LB", "nat": "TR", "age": 25, "att": 0.06, "def": 0.16, "ovr": 71, "starter": True, "val": 400000},
            {"name": "Sefa Akgun", "pos": "DM", "nat": "TR", "age": 25, "att": 0.08, "def": 0.18, "ovr": 72, "starter": True, "val": 600000},
            {"name": "Estrela", "pos": "CM", "nat": "AO", "age": 30, "att": 0.10, "def": 0.18, "ovr": 73, "starter": True, "val": 700000},
            {"name": "Furkan Ozhan", "pos": "AM", "nat": "TR", "age": 24, "att": 0.16, "def": 0.06, "ovr": 72, "starter": True, "val": 600000},
            {"name": "Suleyman Koc", "pos": "RW", "nat": "TR", "age": 36, "att": 0.16, "def": 0.08, "ovr": 71, "starter": True, "val": 250000},
            {"name": "Odise Roshi", "pos": "LW", "nat": "AL", "age": 34, "att": 0.18, "def": 0.06, "ovr": 73, "starter": True, "val": 400000},
            {"name": "Eren Tozlu", "pos": "CF", "nat": "TR", "age": 35, "att": 0.24, "def": 0.01, "ovr": 74, "starter": True, "val": 450000},
            {"name": "Celal Hanalp", "pos": "LB", "nat": "TR", "age": 29, "att": 0.08, "def": 0.14, "ovr": 71, "starter": False, "val": 350000},
        ],
        "synergies": [
            ("defense", "Mustafa Yumlu", "Yakup Kisa", 1.15),
            ("midfield", "Sefa Akgun", "Estrela", 1.16),
            ("attack", "Odise Roshi", "Eren Tozlu", 1.18),
        ]
    },
    35: {  # Kocaelispor
        "name": "Kocaelispor",
        "players": [
            {"name": "Gokhan Degirmenci", "pos": "GK", "nat": "TR", "age": 36, "att": 0.00, "def": 0.20, "ovr": 73, "starter": True, "val": 400000},
            {"name": "Aaron Appindangoye", "pos": "CB", "nat": "GA", "age": 33, "att": 0.02, "def": 0.24, "ovr": 75, "starter": True, "val": 1000000},
            {"name": "Tarkan Serbest", "pos": "CB", "nat": "AT", "age": 31, "att": 0.04, "def": 0.20, "ovr": 73, "starter": True, "val": 700000},
            {"name": "Ahmet Oguz", "pos": "RB", "nat": "TR", "age": 32, "att": 0.10, "def": 0.18, "ovr": 74, "starter": True, "val": 800000},
            {"name": "Muharrem Cinan", "pos": "LB", "nat": "TR", "age": 27, "att": 0.08, "def": 0.16, "ovr": 72, "starter": True, "val": 600000},
            {"name": "Josip Vukovic", "pos": "DM", "nat": "HR", "age": 33, "att": 0.08, "def": 0.22, "ovr": 74, "starter": True, "val": 800000},
            {"name": "Pedrinho Kocaelispor", "pos": "CM", "nat": "BR", "age": 32, "att": 0.20, "def": 0.14, "ovr": 75, "starter": True, "val": 1200000},
            {"name": "Ryan Mendes", "pos": "LW", "nat": "CV", "age": 35, "att": 0.22, "def": 0.06, "ovr": 75, "starter": True, "val": 700000},
            {"name": "Baris Alici", "pos": "RW", "nat": "TR", "age": 28, "att": 0.18, "def": 0.08, "ovr": 73, "starter": True, "val": 800000},
            {"name": "Mijo Caktas", "pos": "AM", "nat": "HR", "age": 33, "att": 0.22, "def": 0.08, "ovr": 75, "starter": True, "val": 1000000},
            {"name": "Markao", "pos": "CF", "nat": "BR", "age": 31, "att": 0.25, "def": 0.01, "ovr": 75, "starter": True, "val": 1500000},
            {"name": "Ahmet Sagat", "pos": "CF", "nat": "TR", "age": 29, "att": 0.16, "def": 0.01, "ovr": 71, "starter": False, "val": 600000},
        ],
        "synergies": [
            ("defense", "Aaron Appindangoye", "Tarkan Serbest", 1.15),
            ("midfield", "Josip Vukovic", "Pedrinho Kocaelispor", 1.18),
            ("attack", "Ryan Mendes", "Markao", 1.20),
            ("attack", "Mijo Caktas", "Markao", 1.18),
        ]
    },
}


def seed_players_database(db_path: str = "data/warehouse.duckdb", json_out: str = "config/squads_2627.json") -> dict[str, int]:
    """Populate players and team_synergy_units tables in DuckDB warehouse and write config/squads_2627.json."""
    con = duckdb.connect(db_path)

    # Ensure tables exist
    con.execute("""
        CREATE TABLE IF NOT EXISTS players (
            player_id INTEGER PRIMARY KEY,
            team_id INTEGER NOT NULL REFERENCES teams(team_id),
            name TEXT NOT NULL,
            position TEXT NOT NULL CHECK (position IN ('GK', 'CB', 'LB', 'RB', 'DM', 'CM', 'AM', 'RW', 'LW', 'CF')),
            nationality TEXT,
            age SMALLINT,
            att_rating DOUBLE NOT NULL DEFAULT 0.0,
            def_rating DOUBLE NOT NULL DEFAULT 0.0,
            overall_rating DOUBLE NOT NULL DEFAULT 70.0,
            market_value_eur BIGINT DEFAULT 1000000,
            is_starter BOOLEAN DEFAULT FALSE
        );
        CREATE TABLE IF NOT EXISTS team_synergy_units (
            team_id INTEGER NOT NULL REFERENCES teams(team_id),
            unit_type TEXT NOT NULL CHECK (unit_type IN ('defense', 'midfield', 'attack')),
            player1_id INTEGER NOT NULL REFERENCES players(player_id),
            player2_id INTEGER NOT NULL REFERENCES players(player_id),
            synergy_score DOUBLE NOT NULL DEFAULT 1.0,
            PRIMARY KEY (team_id, unit_type, player1_id, player2_id)
        );
    """)

    existing_tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
    if "availability_overrides" in existing_tables:
        con.execute("DELETE FROM availability_overrides")
    if "availability_snapshots" in existing_tables:
        con.execute("DELETE FROM availability_snapshots")
    if "player_match_stats" in existing_tables:
        con.execute("DELETE FROM player_match_stats")
    if "player_season_stats" in existing_tables:
        con.execute("DELETE FROM player_season_stats")
    con.execute("DELETE FROM team_synergy_units")
    con.execute("DELETE FROM players")

    player_id_counter = 0
    players_rows = []
    name_to_id: dict[tuple[int, str], int] = {}

    # Canonical pre-season baseline is SQUADS_DATA (frozen, no in-season circular leakage)
    squads_source = SQUADS_DATA
    # Always persist frozen baseline to squads_2627.json
    try:
        Path(json_out).write_text(json.dumps({str(k): v for k, v in SQUADS_DATA.items()}, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

    for tid, sdata in squads_source.items():
        for p in sdata["players"]:
            player_id_counter += 1
            pid = player_id_counter
            name_to_id[(tid, p["name"])] = pid
            players_rows.append((
                pid,
                tid,
                p["name"],
                p["pos"],
                p.get("nat", "TR"),
                p.get("age", 26),
                float(p["att"]),
                float(p["def"]),
                float(p["ovr"]),
                int(p.get("val", 1000000)),
                bool(p.get("starter", False)),
            ))

    con.executemany("""
        INSERT INTO players (player_id, team_id, name, position, nationality, age, att_rating, def_rating, overall_rating, market_value_eur, is_starter)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, players_rows)

    synergy_rows = []
    for tid, sdata in squads_source.items():
        for utype, p1_name, p2_name, score in sdata.get("synergies", []):
            p1_id = name_to_id.get((tid, p1_name))
            p2_id = name_to_id.get((tid, p2_name))
            if p1_id and p2_id:
                synergy_rows.append((tid, utype, p1_id, p2_id, float(score)))

    if synergy_rows:
        con.executemany("""
            INSERT INTO team_synergy_units (team_id, unit_type, player1_id, player2_id, synergy_score)
            VALUES (?, ?, ?, ?, ?)
        """, synergy_rows)

    con.commit()
    con.close()

    # Save to JSON configuration
    out_p = Path(json_out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(squads_source, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "teams_seeded": len(squads_source),
        "players_seeded": len(players_rows),
        "synergies_seeded": len(synergy_rows),
    }


if __name__ == "__main__":
    res = seed_players_database()
    print(f"Seeded {res['players_seeded']} players across {res['teams_seeded']} teams with {res['synergies_seeded']} unit synergies.")
