"""Ingest authentic 2026/27 Süper Lig squads, statistics, and ratings from Mackolik (arsiv.mackolik.com)."""

from datetime import datetime
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any, Optional
import urllib.request
import duckdb
import numpy as np

# Canonical mapping of Süper Lig teams in warehouse.duckdb to Mackolik Team IDs
MACKOLIK_TEAMS: dict[int, dict[str, Any]] = {
    1: {"name": "Galatasaray", "mid": 1},
    2: {"name": "Fenerbahce", "mid": 2},
    3: {"name": "Besiktas", "mid": 3},
    4: {"name": "Trabzonspor", "mid": 4},
    5: {"name": "Basaksehir", "mid": 451},
    6: {"name": "Konyaspor", "mid": 447},
    8: {"name": "Kasimpasa", "mid": 656},
    10: {"name": "Alanyaspor", "mid": 619},
    12: {"name": "Gaziantep FK", "mid": 572},
    15: {"name": "Rizespor", "mid": 448},
    17: {"name": "Goztepe", "mid": 15},
    20: {"name": "Samsunspor", "mid": 8},
    26: {"name": "Eyupspor", "mid": 557},
    27: {"name": "Genclerbirligi", "mid": 7},
    30: {"name": "Erzurumspor FK", "mid": 9632},
    35: {"name": "Kocaelispor", "mid": 18},
    38: {"name": "Corum FK", "mid": 9665},
    39: {"name": "Amedspor", "mid": 591},
}

# Natural position hints for known players across all 18 Süper Lig teams to guarantee authentic tactical roles
KNOWN_PLAYER_POSITIONS: dict[str, str] = {
    # 1. Galatasaray
    "Uğurcan Çakır": "GK", "Jankat Yılmaz": "GK", "Günay Güvenç": "GK", "Arda Yılmaz": "GK", "Enes Emre Büyük": "GK",
    "Davinson Sanchez": "CB", "Abdülkerim Bardakcı": "CB", "Bitshiabu": "CB", "Arda Ünyay": "CB",
    "Kaan Bayri": "CB", "Ali Emir Esen": "CB", "Abdulsamet Şimşek": "CB",
    "Ismail Jakobs": "LB", "Eren Elmalı": "LB", "Kazımcan Karataş": "LB",
    "Roland Sallai": "RB", "Wilfried Singo": "RB", "Kaan Ayhan": "RB",
    "Lucas Torreira": "DM", "Mario Lemina": "DM", "Lesley Ugochukwu": "DM", "Eyüp Aydın": "DM",
    "Gabriel Sara": "CM", "İlkay Gündoğan": "CM", "Renato Nhaga": "CM", "İlhami Siraçhan Nas": "CM",
    "Aleksey Batrakov": "AM", "Onur Kağan Yıldız": "AM", "Ada Karatepe": "AM",
    "Yunus Akgün": "RW", "Leroy Sane": "RW", "Can Armando Güner": "RW",
    "Barış Alper Yılmaz": "LW", "Rafael Leao": "LW", "Berat Luş": "LW",
    "Victor Osimhen": "CF", "Deniz Gül": "CF", "Ada Yüzgeç": "CF", "Efe Gündal": "CF", "Arda Tagay": "CF", "Abdullah Kolazlı": "CF",

    # 2. Fenerbahçe
    "Ederson": "GK", "Mert Günok": "GK", "Tarık Çetin": "GK", "Doruk Yazıcı": "GK",
    "Milan Skriniar": "CB", "Nathan Ake": "CB", "Kojo Peprah Oppong": "CB", "Yiğit Efe Demir": "CB",
    "Rodrigo Becao": "CB", "Alexander Djiku": "CB", "Gökmen Özdemir": "CB",
    "Archie Brown": "LB", "Jayden Oosterwolde": "LB", "Levent Mercan": "LB",
    "Nelson Semedo": "RB", "Mert Müldür": "RB", "Ognjen Mimovic": "RB", "Bright Osayi-Samuel": "RB",
    "Matteo Guendouzi": "DM", "İsmail Yüksek": "DM", "Bartuğ Elmaz": "DM", "Sofyan Amrabat": "DM", "Abdou Aziz Fall": "DM",
    "NGolo Kante": "CM", "Mert Hakan Yandaş": "CM", "Fred": "CM", "Efe Yılmaz": "CM", "Baran Yolaçan": "CM",
    "İrfan Can Kahveci": "AM", "Marco Asensio": "AM", "Sebastian Szymanski": "AM", "Adnan Efe Fettahoğlu": "AM",
    "Mason Greenwood": "RW", "Oğuz Aydın": "RW", "Amara Diouf": "RW",
    "Kerem Aktürkoğlu": "LW", "Dorgeles Nene": "LW", "Yiğit Çolak": "LW", "Alaettin Ekici": "LW",
    "Vedat Muriqi": "CF", "Romelu Lukaku": "CF", "Youssef En-Nesyri": "CF", "Cenk Tosun": "CF",
    "Çağrı Hakan Balta": "CF", "Selahattin Sevim": "CF", "Onur Kuşcu": "CF", "Ercüment Sancaklı": "CF",

    # 3. Beşiktaş
    "Alexander Nübel": "GK", "Doğan Alemdar": "GK", "Emir Yaşar": "GK", "Ersin Destanoğlu": "GK",
    "Emirhan Topçu": "CB", "Tiago Djalo": "CB", "Emmanuel Agbadou": "CB", "Ümit Akdağ": "CB",
    "Yasin Özcan": "CB", "Cumali Gürsel": "CB", "Gabriel Paulista": "CB", "Felix Uduokhai": "CB",
    "Rıdvan Yılmaz": "LB", "Kassoum Ouattara": "LB", "Emrecan Uzunhan": "LB", "Arda Berk Özüarap": "LB",
    "Amir Murillo": "RB", "Taylan Bulut": "RB", "Azem Yortaç": "RB", "Jonas Svensson": "RB",
    "Salih Özcan": "DM", "Wilfred Ndidi": "DM", "Kartal Yılmaz": "DM", "Fahri Ay": "DM", "Al-Musrati": "DM",
    "Orkun Kökçü": "CM", "Fabio Miretti": "CM", "Gedson Fernandes": "CM",
    "Junior Olaitan": "AM", "Rafa Silva": "AM",
    "Vaclav Cerny": "RW", "Ernest Poku": "RW", "Milot Rashica": "RW", "Can Keleş": "RW", "Ozan Sevim": "RW",
    "Leandro Trossard": "LW", "İlhan Fakılı": "LW",
    "Hyeon-Gyu Oh": "CF", "Dusan Vlahovic": "CF", "Semih Kılıçsoy": "CF",
    "Mustafa Hekimoğlu": "CF", "Ahmet Sami Bircan": "CF", "Tammy Abraham": "CF",

    # 4. Trabzonspor
    "Andre Onana": "GK", "Ahmet Doğan Yıldırım": "GK", "Onuralp Çevikkan": "GK", "Erol Can Çolak": "GK",
    "Stefan Savic": "CB", "Chibuike Nwaiwu": "CB", "Cenk Özkacar": "CB", "Samet Akaydin": "CB",
    "Arsenii Batagov": "CB", "Arseniy Batagov": "CB", "Mustafa Kuzey Akçay": "CB", "Murat Lakot": "CB",
    "Ferhat Usta": "CB", "Esad Turan Uzun": "CB", "Alperen Çakıroğlu": "CB",
    "Mustafa Eskihellaç": "LB", "Sidny Lopes Cabral": "LB", "Borna Barisic": "LB",
    "Wagner Pina": "RB", "Pedro Malheiro": "RB",
    "Fabinho": "DM", "Batista Mendy": "DM", "Okay Yokuşlu": "DM", "John Lundstram": "DM",
    "Melih Kabasakal": "CM", "Ozan Tufan": "CM", "Benjamin Bouchouari": "CM", "Tim Folcarelli": "CM",
    "Yiğithan Çubukcu": "CM", "Oğuzhan Ertem": "CM",
    "Ernest Muçi": "AM", "Ruslan Malinovskyi": "AM", "Muhammed Cham": "AM",
    "Mohamed Salah": "RW", "Noah Saviolo": "RW", "Cihan Çanak": "RW",
    "Aral Şimşir": "LW", "Metehan Mimaroğlu": "LW", "Anthony Nwakaeme": "LW", "Denis Dragus": "LW",
    "Paul Onuachu": "CF", "Franculino": "CF", "Umut Nayir": "CF",
    "Ahmet Güren": "CF", "Hakan Yilmaz": "CF", "Ali Habeşoğlu": "CF",

    # 5. Başakşehir
    "Muhammed Şengezer": "GK", "Deniz Dilmen": "GK", "Volkan Babacan": "GK", "Ege Öztürk": "GK", "Lucas Stancic": "GK",
    "Emin Bayram": "CB", "Jerome Opoku": "CB", "Ousseynou Ba": "CB", "Hamza Güreler": "CB",
    "Christopher Operi": "LB", "Michal Karbownik": "LB", "Saba Kharebashvili": "LB",
    "Ömer Ali Şahiner": "RB", "Onur Bulut": "RB",
    "Umut Güneş": "DM", "Jakub Kaluzinski": "DM", "Onur Ergün": "DM",
    "Olivier Kemen": "CM", "Berkay Özcan": "CM",
    "Abbosbek Fayzullayev": "AM",
    "Andreas Skov Olsen": "RW", "Yusuf Sarı": "RW", "Edin Visca": "RW",
    "Ivan Brnic": "LW",
    "Eldor Shomurodov": "CF", "Bertuğ Yıldırım": "CF", "Davie Selke": "CF", "Umut Bozok": "CF", "Batuhan Çelik": "CF",

    # 6. Konyaspor
    "Bahadır Güngördü": "GK", "Deniz Ertaş": "GK", "Egemen Aydın": "GK", "Emir Gündoğdu": "GK",
    "Adil Demirbağ": "CB", "Uğurcan Yazğılı": "CB", "Rayyan Baniya": "CB", "Chidozie Awaziem": "CB",
    "Ata Yanık": "CB", "Joao Pedro do Nascimento da Mata": "CB", "Eric Konate": "CB",
    "Arthur Masuaku": "LB", "Arif Boşluk": "LB",
    "Yhoan Andzouana": "RB", "Ahmet Oğuz": "RB",
    "Marko Jevtovic": "DM", "Mücahit İbrahimoğlu": "DM",
    "Melih İbrahimoğlu": "CM", "Ömer Çobanoğlu": "CM", "Kaan Akyazı": "CM",
    "Rajmund Toth": "AM", "Enis Bardhi": "AM",
    "Deniz Türüç": "RW", "Ebrima Colley": "RW", "Emir Bars": "RW",
    "Diogo Gonçalves": "LW", "Jean-Luc Dompe": "LW",
    "Jackson Muleka": "CF", "Enis Destan": "CF", "Mostafa Mohamed": "CF", "Eren Yağmur": "CF", "Berke Özçelik": "CF",

    # 8. Kasımpaşa
    "Andreas Gianniotis": "GK", "Ali Emre Yanar": "GK", "Hacı Ramazan Özkanlı": "GK", "Ege Albayrak": "GK",
    "Matei Cristian Ilie": "CB", "Adem Arous": "CB", "Kevin Mouanga": "CB", "Ahmet Salih Arıcı": "CB",
    "Ayberk Karapo": "LB", "Mortadha Ben Ouanes": "LB", "Godfried Frimpong": "LB", "Ömer Bayram": "LB", "Ahmet Taha Dağbaşı": "LB",
    "Claudio Winck": "RB", "Kamil Ahmet Çörekçi": "RB",
    "Andri Baldursson": "DM", "Jakob Jessen": "DM", "Atakan Müjde": "DM",
    "Kerem Demirbay": "CM", "Elson Mendes": "CM", "Emirhan Boz": "CM", "Sinan Alkaş": "CM",
    "Marcus Rafferty": "AM", "Haris Hajradinovic": "AM",
    "Ali Yavuz Kol": "RW", "Jesuran Rak-Sakyi": "RW", "Fousseni Diabate": "RW", "Mamadou Fall": "RW",
    "Adrian Benedyczak": "LW", "Thiemoko Diarra": "LW",
    "Güven Yalçın": "CF", "Sercan Deniz": "CF", "Muhammed Emin Bektaş": "CF", "Emirhan Yiğit": "CF",

    # 10. Alanyaspor
    "Paulo Victor": "GK", "Mert Furkan Bayram": "GK", "Yusuf Karagöz": "GK", "Mahmut Can Kara": "GK",
    "Nuno Lima": "CB", "Fatih Aksoy": "CB", "Bedirhan Özyurt": "CB", "Bilal Demirağ": "CB",
    "Fidan Aliti": "LB", "Yusuf Özdemir": "LB", "Batuhan Yavuz": "LB",
    "Florent Hadergjonaj": "RB", "Şahin Dik": "RB",
    "Gaius Makouta": "DM", "Maestro": "DM", "İzzet Çelik": "DM", "Arouna Kigbafory Soro": "DM",
    "Baran Ali Gezek": "CM", "Enes Keskin": "CM", "Yusuf Can Karademir": "CM", "Nejdet Nezir Bilin": "CM",
    "İbrahim Kaya": "AM", "Ianis Hagi": "AM", "Emre Demir": "AM",
    "Ruan": "RW", "Elia Meschack": "RW",
    "Arda Usluoğlu": "LW", "Paulo Fernandes": "LW", "Omar Ben Ali": "LW",
    "Ui-jo Hwang": "CF", "Ivan Cedric": "CF", "Veysel Ünal": "CF",

    # 12. Gaziantep FK
    "Kacper Tobiasz": "GK", "Ataberk Dadakdeniz": "GK", "Ibrahim Kağan Alkış": "GK", "Cemilhan Aslan": "GK",
    "Myenty Abena": "CB", "Ulrich Meleke": "CB", "Arda Kızıldağ": "CB", "Abakar Sylla": "CB",
    "Efe Sarıkaya": "CB", "Mervan Müjdeci": "CB",
    "Kerim Çalhanoğlu": "LB", "Florin Stefan": "LB",
    "Luis Perez": "RB", "Nazım Sangare": "RB", "Sabahattin Destici": "RB",
    "Victor Gidado": "DM", "Ogün Özçiçek": "DM", "Karamba Gassama": "DM",
    "Kacper Kozlowski": "CM", "Juninho Bacuna": "CM", "Taha Güneş": "CM",
    "Alexandru Maxim": "AM", "Mirza Cihan": "AM", "Muhammet Akmelek": "AM",
    "Deian Sorescu": "RW", "Enver Kulasin": "RW",
    "Drissa Camara": "LW", "Sontje Hansen": "LW",
    "Halil Dervişoğlu": "CF", "Trivante Stewart": "CF", "Serdar Dursun": "CF",
    "Fuat Bavuk": "CF", "Ali Osman Kalın": "CF", "Burak Enes Yıkıcı": "CF",

    # 15. Çaykur Rizespor
    "Yahia Fofana": "GK", "Zafer Görgen": "GK", "Efe Doğan": "GK", "Hasan Yağız Döne": "GK",
    "Modibo Sagnan": "CB", "Attila Mocsi": "CB", "Tayyip Talha Sanuç": "CB",
    "Siaka Bakayoko": "CB", "Husniddin Alikulov": "CB", "Habil Özbakır": "CB",
    "Zakaria Ariss": "LB", "Umut Erdem": "LB", "Muhammed Baltacı": "LB",
    "Mithat Pala": "RB", "Taha Şahin": "RB",
    "Taylan Antalyalı": "DM", "Moussa Diakité": "DM",
    "Qazim Laçi": "CM", "Can Bozdogan": "CM", "Mustafa Tosun": "CM",
    "Ibrahim Olawoyin": "AM", "Dal Varesanovic": "AM",
    "Adedire Mebude": "RW", "Emrecan Bulut": "RW", "Iustin Doicaru": "RW",
    "Ahmed Kutucu": "LW", "Valentin Mihaila": "LW",
    "Ali Sowe": "CF", "Gennaro Borrelli": "CF", "Emirhan Yılmaz": "CF", "Yakup Ayan": "CF",

    # 17. Göztepe
    "Arda Özçimen": "GK", "Luka Gugeshashvili": "GK", "Mehmet Şamil Öztürk": "GK", "Nevzat Üzel": "GK",
    "Taha Altıkardeş": "CB", "Malcom Bokele": "CB", "Ege Yıldırım": "CB", "Allan": "CB",
    "Noah Sonko Sundberg": "CB", "Furkan Bayır": "CB", "Berke Sarigül": "CB",
    "Richard Akonnor": "LB",
    "Arda Okan Kurtulan": "RB", "Ogün Bayrak": "RB",
    "Rhaldney": "DM", "Pah Gouhon": "DM",
    "Novatus Miroshi": "CM", "Alex Matos": "CM",
    "Alexis Antunes": "AM", "Efkan Bekiroğlu": "AM", "Faustino Anjorin": "AM",
    "Janderson": "RW", "Gökdeniz Bayrakdar": "RW", "Guilherme Luiz": "RW",
    "Juan": "LW", "Salem Bouajila": "LW", "Deniz Şeker": "LW",
    "Andre Henrique": "CF", "Sinclair Armstrong": "CF", "Bekir Böke": "CF",
    "Ibrahim Sabra": "CF", "Yusuf Ali Şirin": "CF", "Ömür Turan": "CF",

    # 20. Samsunspor
    "Okan Kocuk": "GK", "Bilal Bayazit": "GK", "Efe Üstün": "GK", "Efe Berat Töruz": "GK",
    "Gabriele Guarino": "CB", "Yunus Emre Çift": "CB", "Toni Borevkovic": "CB", "Strahinja Erakovic": "CB",
    "Ali Badra Diabate": "CB", "Igor Drapinski": "CB", "Bedirhan Çetin": "CB", "Kerem Fidan": "CB",
    "Logi Tomasson": "LB", "Enes Albak": "LB",
    "Joe Mendes": "RB", "Mustafa Tan": "RB",
    "Celil Yüksel": "DM", "Elliot Watt": "DM",
    "Yalçın Kayan": "CM", "Oskar Ohlenschlaeger": "CM",
    "Samed Onur": "AM", "Afonso Sousa": "AM", "Antoine Sekongo": "AM",
    "Emre Kılınç": "RW", "Elayis Tavsan": "RW",
    "Saikuba Jarju": "LW", "Tanguy Coulibaly": "LW", "Kouadou Jaures Assoumou": "LW",
    "Mohamed Bayo": "CF", "Marius Mouandilmadji": "CF", "Fatih Kaya": "CF", "Ebrima Ceesay": "CF",

    # 26. Eyüpspor
    "Horatiu Moldovan": "GK", "Emre Bilgin": "GK", "Umut Keseci": "GK",
    "Zak Jules": "CB", "Jawad El Yamiq": "CB", "Charles-Andre Raux Yao": "CB", "Anıl Yaşar": "CB",
    "Arda Yavuz": "CB", "Yiğit Demir": "CB", "Berhan Kutlay Şatlı": "CB",
    "Simone Giordano": "LB", "Ömer Ahmet Ceyhan": "LB",
    "Talha Ülvan": "RB", "Diabel Ndoye": "RB",
    "Chandrel Massanga": "DM", "Taşkın İlter": "DM", "Christ Otiniel Sadia": "DM",
    "David Costa": "CM", "Hamza Akman": "CM", "Erdem Çalık": "CM", "Berkay Kumlu": "CM",
    "Abdelhamid Sabiri": "AM", "Mete Kaan Demir": "AM",
    "Bilal Boutobba": "RW", "Musa Eren Karakaş": "RW",
    "Konrad Michalak": "LW", "Lenny Pintor": "LW", "Prince Obeng Ampem": "LW",
    "Ahmed Abdullahi": "CF", "Yusuf Barası": "CF", "Abdou Sy": "CF",

    # 27. Gençlerbirliği
    "İrfan Can Eğribayat": "GK", "Gökhan Akkan": "GK", "Berk Deniz Çukurcu": "GK",
    "Dimitris Goutas": "CB", "Thalisson": "CB", "Metehan Baltacı": "CB",
    "Yiğit Hamza Aydar": "CB", "Muhammet Çavuşoğlu": "CB",
    "Abdurrahim Dursun": "LB", "Kevin Rodrigues": "LB",
    "Pedro Pereira": "RB", "Fıratcan Üzüm": "RB",
    "Ousmane Diabate": "DM", "Cheikh Niasse": "DM", "Peter Etebo": "DM", "Ensar Kemaloğlu": "DM",
    "Oğulcan Ülgün": "CM", "Rafael Luis": "CM", "Salih Uçan": "CM", "Zafer Göktuğ Erdem": "CM",
    "Franco Tongya": "AM", "Arda Çağan Çelik": "AM", "Erk Arda Aslan": "AM",
    "Tiago Gouveia": "RW", "Prince Martor": "RW",
    "Adama Traore": "LW", "Dilhan Demir": "LW",
    "Sekou Koita": "CF", "Pedro Mendes": "CF", "Victor Orakpo": "CF",
    "Furkan Özcan": "CF", "Hasan Oğul": "CF", "Arda Akgül": "CF",

    # 30. Erzurumspor FK
    "Ertuğrul Taşkıran": "GK", "Matija Orbanic": "GK", "Erkan Anapa": "GK", "Şerefhan Sağlık": "GK",
    "Amar Gerxhaliu": "CB", "Nihad Mujakic": "CB", "Mustafa Yumlu": "CB", "Yakup Kırtay": "CB",
    "Ömer Kara": "CB", "Emre Erdem": "CB", "Taha Rençber": "CB",
    "Guram Giorbelidze": "LB", "Cengizhan Bayrak": "LB",
    "Orhan Ovacıklı": "RB", "Festy Ebosele": "RB", "Yiğit Baran Karaoğlan": "RB",
    "Elisha Owusu": "DM", "Lawrence Agyekum": "DM", "Mert Önal": "DM",
    "Brandon Baiye": "CM", "Sefa Akgün": "CM", "Murat Cem Akpınar": "CM", "Eren Özdemir": "CM", "Emirhan Acar": "CM",
    "Martin Rodriguez": "AM", "Furkan Özhan": "AM",
    "Mustafa Fettahoğlu": "RW", "Gyrano Kerk": "RW", "Neriman Axundzade": "RW",
    "Miguel Cardoso": "LW",
    "Eren Tozlu": "CF", "Ibrahim Yalatif Diabate": "CF", "Fernando": "CF", "İlkan Sever": "CF",

    # 35. Kocaelispor
    "Serhat Öztaşdelen": "GK", "Aleksandar Jovanovic": "GK", "Onurcan Piri": "GK", "Talha Polat": "GK",
    "Tanguy Zoukrou": "CB", "Emir Ortakaya": "CB", "Matej Maglica": "CB", "Onur Öztonga": "CB",
    "Mikdat Çil": "CB", "Deniz Ceylan": "CB",
    "Massadio Haidara": "LB", "Muharrem Cinan": "LB",
    "Anfernee Dijksteel": "RB", "Uğur Kaan Yıldız": "RB", "Tayfur Bingöl": "RB",
    "Show": "DM", "Habib Keita": "DM", "Hulusi Destici": "DM",
    "Berkan Kutlu": "CM", "Bedirhan Yıldız": "CM", "Ege Bilim": "CM",
    "Mahamadou Sissoho": "AM", "Tobias Gulliksen": "AM", "Haydar Karataş": "AM",
    "Gonçalo Sousa": "RW", "Rigoberto Rivas": "RW", "Arda Özyar": "RW",
    "Makana Baku": "LW", "Metehan Altunbaş": "LW",
    "Daniel Agyei": "CF", "Florian Aye": "CF", "Bruno Petkovic": "CF",

    # 38. Çorum FK
    "Marcos Felipe": "GK", "Erhan Erentürk": "GK", "Arif Şimşir": "GK",
    "Hrvoje Smolcic": "CB", "Alexandre Penetra": "CB", "Serdar Saatçı": "CB",
    "Çağlar Söyüncü": "CB", "Arda Hilmi Şengül": "CB",
    "Andrei Borza": "LB", "Cemali Sertel": "LB",
    "Gökhan Sazdağı": "RB",
    "Ylber Ramadani": "DM", "Berat Özdemir": "DM", "Atakan Akkaynak": "DM",
    "Mohammed Diomande": "CM", "Ahmed Ildız": "CM", "Furkan Çetinkaya": "CM", "Semih Akyıldız": "CM",
    "Fredy": "AM", "Markus Karlsbakk": "AM", "Ermin Mahmic": "AM", "Göktan Gürpüz": "AM",
    "Cengiz Ünder": "RW", "Hasan Abdulkareem Sayyid": "RW", "Kenan Fakılı": "RW",
    "Alexandros Kiziridis": "LW", "Emircan Gürlük": "LW", "Hüseyin Bulut": "LW",
    "Jesus Ramirez": "CF", "Youssoufa Moukoko": "CF", "Mame Thiam": "CF", "Hasan Ege Akdoğan": "CF",

    # 39. Amedspor
    "Alban Lafont": "GK", "Burak Bozan": "GK", "Veysel Sapan": "GK", "Muhammed Damar": "GK",
    "David Bates": "CB", "Lumbardh Dellova": "CB", "Mehmet Yeşil": "CB", "Kahraman Demirtaş": "CB",
    "Umut Meraş": "LB", "Miraç Acer": "LB",
    "Cem Üstündağ": "RB", "Ali Turap Bülbül": "RB",
    "Rayan Raveloson": "DM", "Gökhan Gül": "DM", "Amadou Cisse": "DM",
    "Furkan Soyalp": "CM", "Mohamed Khalil": "CM", "Rayan Lutin": "CM", "Berat Perçin": "CM",
    "Dia Sabia": "AM", "Muhammed Hasan Yıldırım": "AM", "Berk Kızıldemir": "AM",
    "Samuel Ballet": "RW", "Yira Collins Sor": "RW",
    "Ermal Krasniqi": "LW",
    "Gift Orban": "CF", "Mbaye Diagne": "CF", "İsmail Duyu": "CF",
}

HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://arsiv.mackolik.com/",
    "Accept": "application/json, text/javascript, */*; q=0.01",
}


def parse_market_value(val_str: Any) -> int:
    """Parse market value strings like '16.000.000', '1.750.000', '50.000' or '-' into integer EUR."""
    if not val_str:
        return 750000
    s = str(val_str).strip().replace(".", "").replace(" ", "").replace("€", "")
    if not s or s == "-":
        return 750000
    try:
        val = int(s)
        return max(50000, val)
    except ValueError:
        return 750000


def fetch_mackolik_team_squad(
    mackolik_team_id: int,
    season: str = "2026/2027",
    retries: int = 3,
    delay: float = 0.5,
) -> dict[str, Any]:
    """Fetch raw squad JSON for a given team from Mackolik."""
    url = f"https://arsiv.mackolik.com/Team/SquadData.aspx?id={mackolik_team_id}&season={season}"
    req = urllib.request.Request(url, headers=HTTP_HEADERS)

    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=12) as resp:
                raw_bytes = resp.read()
                return json.loads(raw_bytes.decode("utf-8"))
        except Exception as e:
            if attempt == retries - 1:
                raise RuntimeError(f"Failed to fetch Mackolik squad for team {mackolik_team_id}: {e}") from e
            time.sleep(delay * (attempt + 1))
    return {}


def parse_team_squad(
    team_id: int,
    team_name: str,
    raw_squad: dict[str, Any],
) -> dict[str, Any]:
    """
    Parse Mackolik raw squad data, assign positions, starter status,
    ratings, and tactical unit synergies.
    """
    # Build country lookup table
    countries = {str(c[0]): str(c[2]) for c in raw_squad.get("c", [])}
    countries["1"] = "TR"

    parsed_players = []
    for sec in raw_squad.get("s", []):
        cat = sec.get("p")  # 'K', 'D', 'O', 'F'
        for p in sec.get("s", []):
            jersey = str(p[0])
            mackolik_pid = str(p[1])
            name = str(p[2]).strip()
            cid = str(p[3])
            nat = countries.get(cid, "TR")[:3]
            age = int(p[4]) if str(p[4]).isdigit() else 25
            matches = int(p[5]) if str(p[5]).isdigit() else 0
            assists = int(p[6]) if str(p[6]).isdigit() else 0
            starts = int(p[7]) if str(p[7]).isdigit() else 0
            mins = int(p[8]) if str(p[8]).isdigit() else 0
            goals = int(p[9]) if str(p[9]).isdigit() else 0
            yellow = int(p[10]) if str(p[10]).isdigit() else 0
            red = int(p[11]) if str(p[11]).isdigit() else 0
            val = parse_market_value(p[12])

            parsed_players.append({
                "jersey": jersey,
                "mackolik_id": mackolik_pid,
                "name": name,
                "cat": cat,
                "nat": nat,
                "age": age,
                "matches": matches,
                "starts": starts,
                "minutes": mins,
                "goals": goals,
                "assists": assists,
                "yellow": yellow,
                "red": red,
                "val": val,
            })

    # Sort all players by starter merit: starts first, then minutes, matches, and market value
    merit_key = lambda x: (x["starts"], x["minutes"], x["matches"], x["val"])
    parsed_players.sort(key=merit_key, reverse=True)

    # Pass 1: Assign authentic football position to every player FIRST
    pos_counts: dict[str, int] = {
        "GK": 0, "CB": 0, "LB": 0, "RB": 0,
        "DM": 0, "CM": 0, "AM": 0, "RW": 0, "LW": 0, "CF": 0,
    }
    for p in parsed_players:
        known = KNOWN_PLAYER_POSITIONS.get(p["name"])
        if known in pos_counts:
            pos_counts[known] += 1

    def_rot_idx, mid_rot_idx, fwd_rot_idx = 0, 0, 0
    def_rot = ["CB", "LB", "RB", "CB"]
    mid_rot = ["CM", "DM", "AM"]
    fwd_rot = ["CF", "RW", "LW"]

    for p in parsed_players:
        known = KNOWN_PLAYER_POSITIONS.get(p["name"])
        if known in pos_counts:
            p["pos"] = known
            continue

        cat = p["cat"]
        if cat == "K":
            pos = "GK"
        elif cat == "D":
            if pos_counts["CB"] < 2:
                pos = "CB"
            elif pos_counts["LB"] < 1:
                pos = "LB"
            elif pos_counts["RB"] < 1:
                pos = "RB"
            else:
                pos = def_rot[def_rot_idx % len(def_rot)]
                def_rot_idx += 1
        elif cat == "O":
            if pos_counts["DM"] < 1:
                pos = "DM"
            elif pos_counts["CM"] < 1:
                pos = "CM"
            elif pos_counts["AM"] < 1:
                pos = "AM"
            else:
                pos = mid_rot[mid_rot_idx % len(mid_rot)]
                mid_rot_idx += 1
        else:
            if pos_counts["CF"] < 1:
                pos = "CF"
            elif pos_counts["RW"] < 1:
                pos = "RW"
            elif pos_counts["LW"] < 1:
                pos = "LW"
            else:
                pos = fwd_rot[fwd_rot_idx % len(fwd_rot)]
                fwd_rot_idx += 1

        p["pos"] = pos
        pos_counts[pos] += 1

    # Pass 2: Select the 11 starters strictly by tactical formation slots (4-2-3-1 / 4-3-3)
    # and starter merit (who started the most matches and played the most minutes in that position)
    FORMATION_SLOTS: list[tuple[str, int]] = [
        ("GK", 1),
        ("RB", 1),
        ("CB", 2),
        ("LB", 1),
        ("DM", 1),
        ("CM", 1),
        ("AM", 1),
        ("RW", 1),
        ("LW", 1),
        ("CF", 1),
    ]
    by_pos: dict[str, list[dict[str, Any]]] = {pos: [] for pos, _ in FORMATION_SLOTS}
    for p in parsed_players:
        by_pos[p["pos"]].append(p)
    for pos in by_pos:
        by_pos[pos].sort(key=merit_key, reverse=True)

    starter_names: set[str] = set()
    ordered_starters: list[tuple[dict[str, Any], str, bool]] = []

    for pos, target_count in FORMATION_SLOTS:
        candidates = by_pos.get(pos, [])
        selected = candidates[:target_count]
        for p in selected:
            starter_names.add(p["name"])
            ordered_starters.append((p, pos, True))

    # Fallback if a squad had fewer candidates in a sub-slot (fill from same line without duplicating GK or CF)
    if len(ordered_starters) < 11:
        remaining = [
            p for p in parsed_players
            if p["name"] not in starter_names and p["pos"] not in ("GK", "CF")
        ]
        remaining.sort(key=merit_key, reverse=True)
        for p in remaining[: 11 - len(ordered_starters)]:
            starter_names.add(p["name"])
            ordered_starters.append((p, p["pos"], True))

    ordered_bench: list[tuple[dict[str, Any], str, bool]] = [
        (p, p["pos"], False)
        for p in sorted(parsed_players, key=merit_key, reverse=True)
        if p["name"] not in starter_names
    ]

    assigned: list[tuple[dict[str, Any], str, bool]] = ordered_starters + ordered_bench

    final_players = []
    starters_dict: dict[str, list[dict[str, Any]]] = {
        "GK": [], "CB": [], "LB": [], "RB": [],
        "DM": [], "CM": [], "AM": [], "CF": [], "RW": [], "LW": []
    }

    for p, pos, is_st in assigned:
        val = p["val"]
        goals = p["goals"]
        assists = p["assists"]
        starts = p["starts"]
        mins = p["minutes"]

        # OVR Calculation: Logarithmic valuation curve bounded between 68.0 and 87.5
        log_val = math.log10(max(100000, val))
        base_ovr = 65.0 + (log_val - 5.0) * (85.0 - 65.0) / (7.875 - 5.0)
        ovr = base_ovr + (2.0 if is_st else 0.0) + min(3.0, goals * 0.35 + assists * 0.25)
        ovr = round(float(np.clip(ovr, 68.0, 87.5)), 1)

        # ATT Rating Calculation
        if pos == "GK":
            att = 0.00
        elif pos in ("CB", "LB", "RB"):
            att = 0.03 + (0.05 if pos in ("LB", "RB") else 0.0) + min(0.12, goals * 0.04 + assists * 0.03)
        elif pos == "DM":
            att = 0.10 + (ovr - 70.0) * 0.008 + min(0.08, goals * 0.03 + assists * 0.02)
        elif pos == "CM":
            att = 0.16 + (ovr - 70.0) * 0.010 + min(0.10, goals * 0.03 + assists * 0.02)
        elif pos == "AM":
            att = 0.22 + (ovr - 70.0) * 0.012 + min(0.12, goals * 0.03 + assists * 0.02)
        elif pos in ("RW", "LW"):
            att = 0.24 + (ovr - 70.0) * 0.012 + min(0.14, goals * 0.03 + assists * 0.02)
        else:  # CF
            att = 0.28 + (ovr - 70.0) * 0.015 + min(0.18, goals * 0.035 + assists * 0.02)
        att = round(float(np.clip(att, 0.0, 0.46)), 2)

        # DEF Rating Calculation
        if pos == "GK":
            def_r = 0.24 + (ovr - 70.0) * 0.012 + (0.06 if is_st else 0.0)
        elif pos == "CB":
            def_r = 0.26 + (ovr - 70.0) * 0.012 + (0.04 if is_st else 0.0)
        elif pos in ("LB", "RB"):
            def_r = 0.18 + (ovr - 70.0) * 0.010 + (0.03 if is_st else 0.0)
        elif pos == "DM":
            def_r = 0.20 + (ovr - 70.0) * 0.010 + (0.04 if is_st else 0.0)
        elif pos == "CM":
            def_r = 0.14 + (ovr - 70.0) * 0.008
        elif pos == "AM":
            def_r = 0.06 + (ovr - 70.0) * 0.005
        elif pos in ("RW", "LW"):
            def_r = 0.05 + (ovr - 70.0) * 0.004
        else:  # CF
            def_r = 0.02
        def_r = round(float(np.clip(def_r, 0.0, 0.40)), 2)

        p_dict = {
            "name": p["name"],
            "pos": pos,
            "nat": p["nat"],
            "age": p["age"],
            "att": att,
            "def": def_r,
            "ovr": ovr,
            "starter": is_st,
            "val": val,
            "matches": p["matches"],
            "starts": p["starts"],
            "minutes": p["minutes"],
            "goals": p["goals"],
            "assists": p["assists"],
            "yellow": p["yellow"],
            "red": p["red"],
            "mackolik_id": p.get("mackolik_id", ""),
        }
        final_players.append(p_dict)
        if is_st:
            starters_dict[pos].append(p_dict)

    # Tactical Unit Synergies between key starters
    synergies = []
    cbs = starters_dict.get("CB", [])
    gks_st = starters_dict.get("GK", [])
    if len(cbs) >= 2:
        synergies.append(("defense", cbs[0]["name"], cbs[1]["name"], 1.20))
    if gks_st and cbs:
        synergies.append(("defense", gks_st[0]["name"], cbs[0]["name"], 1.18))
        if len(cbs) >= 2:
            synergies.append(("defense", gks_st[0]["name"], cbs[1]["name"], 1.16))

    dms = starters_dict.get("DM", [])
    cms = starters_dict.get("CM", [])
    ams = starters_dict.get("AM", [])
    if dms and cms:
        synergies.append(("midfield", dms[0]["name"], cms[0]["name"], 1.20))
    if cms and ams:
        synergies.append(("midfield", cms[0]["name"], ams[0]["name"], 1.16))
    elif dms and ams:
        synergies.append(("midfield", dms[0]["name"], ams[0]["name"], 1.15))

    cfs = starters_dict.get("CF", [])
    rws = starters_dict.get("RW", [])
    lws = starters_dict.get("LW", [])
    if ams and cfs:
        synergies.append(("attack", ams[0]["name"], cfs[0]["name"], 1.24))
    if rws and cfs:
        synergies.append(("attack", rws[0]["name"], cfs[0]["name"], 1.20))
    if lws and cfs:
        synergies.append(("attack", lws[0]["name"], cfs[0]["name"], 1.18))

    return {
        "name": team_name,
        "players": final_players,
        "synergies": synergies,
    }


def fetch_and_sync_all_mackolik_squads(
    db_path: str = "data/warehouse.duckdb",
    json_out: str = "config/squads_2627.json",
    raw_cache_path: str = "data/raw/mackolik_squads_2627_raw.json",
    use_cache_on_fail: bool = True,
    prefer_cache: bool = False,
) -> dict[str, Any]:
    """
    Fetch 2026/27 squads from Mackolik for all 18 Süper Lig teams,
    update DuckDB warehouse, write config JSON, and recalculate ratings.
    """
    raw_cache_file = Path(raw_cache_path)
    raw_cache_file.parent.mkdir(parents=True, exist_ok=True)

    fetched_raw: dict[int, dict[str, Any]] = {}
    errors: list[str] = []

    if prefer_cache and raw_cache_file.exists():
        try:
            cached_data = json.loads(raw_cache_file.read_text(encoding="utf-8"))
            for tid_str, raw in cached_data.items():
                fetched_raw[int(tid_str)] = raw
        except Exception as e:
            errors.append(f"Cache read error: {e}")

    # Attempt fetching live from Mackolik for any team not loaded from cache
    if len(fetched_raw) < len(MACKOLIK_TEAMS):
        for tid, info in MACKOLIK_TEAMS.items():
            if tid in fetched_raw:
                continue
            try:
                raw = fetch_mackolik_team_squad(info["mid"], season="2026/2027")
                fetched_raw[tid] = raw
                time.sleep(0.15)  # Politeness delay
            except Exception as e:
                errors.append(f"Team {info['name']} (ID {info['mid']}): {e}")

    # If some failed and cache exists, merge from cache
    if errors and use_cache_on_fail and raw_cache_file.exists():
        try:
            cached_data = json.loads(raw_cache_file.read_text(encoding="utf-8"))
            for tid_str, raw in cached_data.items():
                tid = int(tid_str)
                if tid not in fetched_raw:
                    fetched_raw[tid] = raw
        except Exception:
            pass

    # Save cache if we got all or most teams
    if len(fetched_raw) >= 16:
        raw_cache_file.write_text(json.dumps(fetched_raw, ensure_ascii=False, indent=2), encoding="utf-8")

    # Parse and structure teams
    parsed_teams: dict[int, dict[str, Any]] = {}
    for tid, info in MACKOLIK_TEAMS.items():
        if tid in fetched_raw:
            parsed_teams[tid] = parse_team_squad(tid, info["name"], fetched_raw[tid])

    if not parsed_teams:
        raise RuntimeError("No team squads could be parsed from Mackolik!")

    # Write config JSON
    json_path = Path(json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(parsed_teams, ensure_ascii=False, indent=2), encoding="utf-8")

    # Seed into DuckDB
    con = duckdb.connect(db_path)
    try:
        tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
        saved_overrides = []
        if "availability_overrides" in tables and "players" in tables:
            saved_overrides = con.execute("""
                SELECT o.override_id, o.team_id, p.name, o.status, o.match_id, o.note, o.created_at, o.expires_at, o.active
                FROM availability_overrides o
                JOIN players p ON o.player_id = p.player_id
            """).fetchall()
            con.execute("DELETE FROM availability_overrides")
        elif "availability_overrides" in tables:
            con.execute("DELETE FROM availability_overrides")

        if "player_match_stats" in tables:
            con.execute("DELETE FROM player_match_stats")
        if "player_season_stats" in tables:
            con.execute("DELETE FROM player_season_stats")
        if "team_synergy_units" in tables:
            con.execute("DELETE FROM team_synergy_units")
        if "players" in tables:
            con.execute("DELETE FROM players")

        player_id_counter = 0
        players_rows = []
        name_to_id: dict[tuple[int, str], int] = {}

        for tid, sdata in parsed_teams.items():
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
                    p.get("age", 25),
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

        if saved_overrides:
            restored_overrides = []
            for ov_id, tid, p_name, st_val, m_id, note, c_at, e_at, act in saved_overrides:
                new_pid = name_to_id.get((int(tid), str(p_name)))
                if new_pid:
                    restored_overrides.append((ov_id, tid, new_pid, st_val, m_id, note, c_at, e_at, act))
            if restored_overrides:
                con.executemany("""
                    INSERT INTO availability_overrides (override_id, team_id, player_id, status, match_id, note, created_at, expires_at, active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, restored_overrides)

        synergy_rows = []
        for tid, sdata in parsed_teams.items():
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
    finally:
        con.close()

    # Regenerate player match stats and Bayesian season stats
    from slmodel.data.player_performance import (
        generate_and_sync_player_match_stats,
        compute_player_season_stats_and_ratings,
    )
    match_res = generate_and_sync_player_match_stats(db_path=db_path, season="2627")
    season_df = compute_player_season_stats_and_ratings(
        db_path=db_path,
        season="2627",
    )

    return {
        "teams_synced": len(parsed_teams),
        "players_synced": len(players_rows),
        "synergies_synced": len(synergy_rows),
        "matches_processed": match_res["matches_processed"],
        "season_records": len(season_df),
        "errors": errors,
    }


if __name__ == "__main__":
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    print("Ingesting authentic 2026/27 squads from Mackolik...")
    res = fetch_and_sync_all_mackolik_squads()
    print(f"Success! Synced {res['players_synced']} players across {res['teams_synced']} teams.")
