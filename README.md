# Caracterització de la demanda del servei d'atenció a passatgers de mobilitat reduïda en un aeroport emprant xarxes neuronals

Aquest repositori conté el codi amb què s'ha desenvolupat un model de xarxa neuronal per a predir el nombre de passatgers de mobilitat reduïda (PMR) en un aeroport. Per confidencialitat de les dades, el preprocessament d'aquestes no hi és inclòs. 


## Organització

El projecte consta de 2 carpetes i 2 arxius principals:

### `dades/`

Conté els arxius d'extensió `.csv` amb les dades de cada vol. 

Aquests han de tenir un número com a nom (e.g., `39.csv`). Això permet treballar amb diferents dades, simplement canviant el número de l'arxiu. Per a cadascun cal crear una funció amb aquest número a l'arxiu `load_data.py`. Aquest repositori només conté el número 39, que és l'utilitzat en el model final.

Cada línia de l'arxiu ha de tenir el següent format `2013/01/01,0.008,112,0.395833333,9,1,MIA,L,P,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,North America,0`, on cada posició representa la següent característica, respectivament: data (yyyy/mm/dd), aerolínia (representada per un número), número de vol, hora (normalitzada), hora (sense els minuts), operació, aeroport, operació, vol fos, preavisos totals, BLND, DEAF, DPNA, WCHC, WCHR, WCHS, MAAS, MEDA, STCR, WCHP, desconegut, preavisos BLND, preavisos DEAF, preavisos DPNA, preavisos WCHC, preavisos WCHR, preavisos WCHS, preavisos MAAS, preavisos MEDA, preavisos STCR, preavisos WCHP, preavisos desconegut, pmrs, país, Schengen. 


### `libraries/`

Conté les llibreries creades per al projecte, incloent-hi:
- `airport_mapping.py`: conversió de l'aeroport en país
- `encode_data.py`: codificació de les dades
- `load_data.py`: càrrega de les dades trobades a la carpeta `dades/`
- `nn_model.py`: creació, validació i prediccions del model de xarxa neuronal
- `utils.py`: llibreria d'ajuda (e.g., per a tenir informació mentre s'executa el codi)

### `kfold.py`

Codi per a validar el model.

### `main.py`

Codi per a obtenir els resultats.

## Dependències

Cal instal·lar les següents llibreries:
- `numpy`
- `pandas`
- `scikit-learn`
- `tensorflow` (inclou Keras)

## Ús

Es recomana treballar amb un IDE. 

Al començament dels següents arxius, hi ha certes variables que es poden modificar per variar els resultats, com per exemple el número de l'arxiu de dades, les èpoques, el nombre de capes ocultes, o el mes del qual es volen obtenir els resultats (gener per defecte). 

Per a obtenir les prediccions, cal executar l'arxiu:
``main.py``


Per a validar el model, cal executar l'arxiu:
``kfold.py``

## Resultats
L'execució de l'arxiu `main.py` genera els arxius `predictions-39.txt` amb les prediccions del mes especificat en el codi i `weights-39.txt` amb els pesos de cada neurona, amb els quals es poden avaluar els resultats.

Un exemple de predicció és `2018,1,0,0.008,0.819444444,0,LHR,1,2,1.0,2.0,2.3566946983337402`, on cada paràmetre indica les següents característiques: any, mes, dia de la setmana, aerolínia, hora, operació, aeroport, vol fos, preavisos, dia, PMRs, predicció.