# MedVision : piloter sa maison par gestes

Une caméra reconnaît des gestes de la main (et, à terme, des signes de la langue des signes),
ou un radar mmWave reconnaît des gestes amples du corps, puis Home Assistant exécute l'action
associée : allumer une lumière, ouvrir un volet, prévenir un proche…

Le projet s'adresse à tout le monde, et en particulier aux personnes âgées, sourdes ou
malentendantes, ou à mobilité réduite.

- **100 % local** : l'analyse se fait sur votre machine, sans cloud.
- **Respect de la vie privée** : aucune image n'est stockée ni transmise. Seuls des points
  de la main (21 coordonnées) sont calculés, et seul le nom du geste reconnu sort du moteur.
- **Vos propres gestes** : 5 exemples suffisent pour apprendre un geste, sans entraînement,
  depuis une interface web intégrée à Home Assistant.
- **Caméra ou radar** : webcam / caméra IP, ou radar mmWave **HLK-LD2450** (aucune image,
  fonctionne dans le noir). Les capteurs se branchent derrière une interface commune.
- **Sécurité** : un *geste d'attention* optionnel doit précéder chaque commande, pour éviter
  les déclenchements involontaires.

> **Statut : v0.3 expérimentale.** Elle reconnaît des **gestes personnalisés** et des **signes
> isolés** que vous enregistrez vous-même. Elle ne traduit **pas** des phrases en LSF. Vos
> retours décideront de la suite : voir [Donner son avis](#donner-son-avis).

## Comment ça marche

```
 Caméra ─► mouvement ─► MediaPipe ─┐  CameraAdapter
 Radar LD2450 ─► trames UART ──────┤  LD2450Adapter      ┌──► Recorder (enregistrement)
 Simulation ───────────────────────┘  SimulatedAdapter   │
                       même interface SensorAdapter ─► Observation
                                                         │
Home Assistant ◄── MQTT (event « lumiere ») ◄── machine à états ◄── comparaison DTW
  automatisations                (attention, anti-rebond)   aux exemples enregistrés
```

Chaque capteur est un *adapter* ([`src/medvision/sensors/`](src/medvision/sensors)) qui
traduit son API propre en `Observation` (un vecteur de features, une présence, des données de
vue en direct). Le moteur, MQTT et l'interface web ne connaissent que cette interface : ajouter
un capteur revient à écrire une classe et la déclarer avec `@register_sensor("nom")`.

| Capteur | `sensor_type` | Ce qu'il reconnaît | Points forts |
|---|---|---|---|
| Webcam USB / caméra IP | `camera` | gestes des mains, signes isolés | précision des doigts |
| Radar HLK-LD2450 (24 GHz) | `ld2450` | gestes amples : s'approcher, reculer, pas de côté, balayage | aucune image, noir complet, présence immobile, nombre de personnes |
| Simulation | `simulation` | rejoue vos gestes enregistrés | tester sans matériel |

## Démarrage rapide (PC avec webcam)

Prérequis : [uv](https://docs.astral.sh/uv/getting-started/installation/) et une webcam.
uv installe lui-même Python 3.12 si besoin.

```bash
git clone https://github.com/VOTRE_COMPTE/medvision.git
cd medvision
uv sync                              # installe les dépendances (MediaPipe, OpenCV…)
uv run medvision download-model      # modèle de détection des mains (~8 Mo)

# Enregistrer deux gestes (5 exemples chacun, avec aperçu de la caméra)
uv run medvision record lumiere --label "Allumer la lumière" --preview
uv run medvision record repos   --label "Main au repos"      --preview
uv run medvision list

# Tester la reconnaissance, sans Home Assistant
uv run medvision run --preview --dry-run

# … ou tout faire depuis le navigateur : http://127.0.0.1:8099
uv run medvision run --web --dry-run
```

### Avec le radar mmWave HLK-LD2450

Reliez le radar à un adaptateur USB-UART **3,3 V** (TX↔RX, 5 V et GND), puis :

```bash
uv run medvision --sensor ld2450 --radar-port /dev/ttyUSB0 run --web --dry-run
```

Le radar peut aussi être déporté : un ESP32 collé au radar et exposant son port série sur le
réseau (ESPHome *stream server*, ser2net…) se lit avec `--radar-port socket://192.168.50.20:6638`.
Le radar suit la personne et mesure ses déplacements **par rapport à l'endroit où elle se
tient** : un geste appris au milieu de la pièce marche aussi près du canapé. Il ne voit pas les
doigts ; préférez des gestes amples d'au moins 30 cm, avec des exemples de 1,5 à 2 s.

Conseils d'enregistrement :

- Faites le geste comme vous le ferez au quotidien, à la distance habituelle.
- Variez légèrement la position et la vitesse d'un exemple à l'autre.
- Enregistrez un geste `repos` (main immobile, ouverte) : il « absorbe » les mouvements
  ordinaires et réduit les faux déclenchements.
- `medvision list` signale un geste aux exemples incohérents. Réenregistrez-le alors avec
  `--replace`.

## Connexion à Home Assistant

### Option A : add-on (Home Assistant OS)

1. **Paramètres › Modules complémentaires › Boutique › ⋮ › Dépôts**, puis ajoutez
   `https://github.com/VOTRE_COMPTE/medvision`.
2. Installez **MedVision Gesture Engine**, choisissez le capteur (`camera` ou `ld2450`) dans
   les options et démarrez-le.
3. Ouvrez **MedVision** dans la barre latérale et enregistrez vos gestes.

L'add-on est distribué par ce dépôt d'add-ons (`repository.yaml`), pas par HACS : HACS ne
distribue que des intégrations et des cartes.

Les identifiants de Mosquitto sont détectés automatiquement. Détails dans
[medvision-addon/DOCS.md](medvision-addon/DOCS.md).

### Option B : sur un PC ou un serveur, vers votre Home Assistant

```bash
cp config.example.yaml config.yaml   # renseignez mqtt_host, mqtt_username, mqtt_password
uv run medvision -c config.yaml run
```

### Option C : Docker

```bash
uv lock && docker compose up --build   # inclut un broker Mosquitto de test
```

Une image multi-architecture (amd64, arm64) est publiée sur `ghcr.io` à chaque version
(voir `.github/workflows/release.yml`).

### Entités créées automatiquement (découverte MQTT)

| Entité | Rôle |
|---|---|
| `event.medvision_salon_geste` | Déclencheur : un événement par geste (`event_type` = nom du geste) |
| `sensor.medvision_salon_dernier_geste` | Dernier geste reconnu |
| `binary_sensor.medvision_salon_main` | Caméra : une main est visible |
| `binary_sensor.medvision_salon_presence` | Radar : quelqu'un est présent (même immobile) |
| `sensor.medvision_salon_targets` / `_distance` | Radar : nombre de personnes, distance (m) |
| `binary_sensor.medvision_salon_attente` | Geste d'attention reconnu, commande attendue |
| `button.medvision_salon_recharger` | Relit les gestes sans redémarrer |

Voir [examples/automations.yaml](examples/automations.yaml) : lumière, volets, alerte d'un
proche, confirmation visuelle pour les personnes sourdes, veilleuse de nuit au radar.

## Configuration

Toutes les options sont documentées dans [config.example.yaml](config.example.yaml). Chacune
peut aussi passer par une variable d'environnement `MEDVISION_<OPTION>`. Les plus utiles :

| Option | Défaut | Effet |
|---|---|---|
| `sensor_type` | `camera` | `camera`, `ld2450` ou `simulation` |
| `camera_source` | `0` | Index de webcam ou URL RTSP (préférez le flux secondaire basse définition) |
| `radar_port` | `/dev/ttyUSB0` | Port série du radar, ou `socket://ip:port` |
| `radar_max_distance` | `4.0` | Distance (m) au-delà de laquelle le radar ignore les personnes |
| `web_enabled` / `web_host` / `web_port` | `false` / `127.0.0.1` / `8099` | Interface web (sans mot de passe : restez en local) |
| `fps` | `15` | Images analysées par seconde |
| `motion_gating` | `true` | Le réseau de neurones ne tourne que s'il y a du mouvement |
| `attention_gesture` | *(vide)* | Geste obligatoire avant une commande |
| `default_threshold` | `0.35` | Tolérance par défaut (chaque geste est ensuite calibré seul) |
| `cooldown` | `1.5` | Pause en secondes après un geste reconnu |
| `rest_gestures` | `repos` | Gestes neutres : reconnus, jamais envoyés à Home Assistant |

## Matériel conseillé pour tester

| Usage | Matériel |
|---|---|
| Essai sur PC | N'importe quel PC récent et une webcam USB |
| Installation permanente | Raspberry Pi 5 (64 bits) ou mini-PC, avec une webcam USB ou une caméra IP RTSP |
| Radar | HLK-LD2450 (~5 €) + adaptateur USB-UART 3,3 V, ou ESP32 en passerelle Wi-Fi |
| Réseau | Idéalement, caméra et serveur sur un VLAN ou un SSID dédié à la domotique |

Le moteur affiche toutes les minutes une ligne `Performance : … img/s, … % analysées`.
Merci de l'indiquer dans vos retours.

## Développement

```bash
uv sync                    # dépendances + outils de dev
uv run pytest              # tests (sans caméra, radar ni MediaPipe : données synthétiques)
uv run ruff check . && uv run ruff format .
```

```
src/medvision/
  config.py      options (YAML, options.json d'add-on, variables d'environnement)
  features.py    normalisation des 21 points de la main, registre des formats de features
  classifier.py  bibliothèque de gestes, DTW, calibration automatique des seuils
  engine.py      machine à états : fenêtre glissante, stabilité, geste d'attention
  store.py       gestes au format JSON (un format par type de capteur)
  sensors/
    base.py        interface SensorAdapter + Observation
    __init__.py    registre et fabrique create_sensor()
    camera.py      adapter caméra (assemble vision.py)
    ld2450.py      protocole du radar HLK-LD2450 + adapter
    simulation.py  capteur simulé
  vision.py      caméra, détection de mouvement, MediaPipe Hand Landmarker
  recorder.py    enregistrement d'un geste, indépendant du capteur
  runtime.py     boucle principale partagée par la CLI, l'add-on et l'interface web
  web.py         serveur de l'interface web (bibliothèque standard) + static/index.html
  mqtt.py        publication et découverte Home Assistant
  addon.py       point d'entrée de l'add-on
  cli.py         commandes sensors / record / list / delete / run / download-model
```

### Ajouter un capteur

```python
from medvision.features import register_format
from medvision.sensors import Observation, SensorAdapter, register_sensor

register_format("mon-capteur-v1", dim=4, label="mon capteur")


@register_sensor("mon_capteur", "Description affichée par `medvision sensors`")
class MonCapteur(SensorAdapter):
    feature_format = "mon-capteur-v1"
    presence_name = "Présence"

    def open(self): ...
    def read(self) -> Observation | None: ...  # une observation, au rythme du capteur
    def close(self): ...
```

## Feuille de route

- [x] v0.1 : gestes personnalisés à une main, MQTT et découverte HA, add-on, Docker
- [x] v0.3 : architecture *adapter* des capteurs, radar mmWave HLK-LD2450, interface web
  d'enregistrement dans Home Assistant (ingress), image Docker publiée sur GHCR
- [ ] Fusion caméra + radar (le radar réveille la caméra, présence confirmée)
- [ ] Gestes à deux mains et posture du corps (détection de chute sans image)
- [ ] Vocabulaire LSF de base, construit **avec** des signeurs et signeuses
- [ ] Portage du moteur sur une caméra embarquée à bas coût (Rockchip RV1106)

## Vie privée et sécurité

- Les images restent en mémoire le temps d'une analyse (environ 1/15 s), puis sont détruites.
  Le radar ne produit aucune image.
- L'interface web n'affiche que des points dessinés, jamais la vidéo. Elle n'a pas de mot de
  passe : dans l'add-on, Home Assistant l'authentifie ; ailleurs, elle n'écoute que sur
  127.0.0.1 par défaut.
- Les fichiers de gestes ne contiennent que des coordonnées normalisées.
- N'installez pas de caméra chez quelqu'un sans son consentement éclairé.
- Ce logiciel n'est **pas** un dispositif médical, ni un système d'alarme certifié. Ne vous
  en servez pas comme unique moyen d'appeler les secours.

## Donner son avis

Ouvrez une [issue « Retour de test »](../../issues/new/choose). Tous les profils
nous intéressent, en particulier les personnes âgées, sourdes ou signeuses, et leurs aidants.

Licence : [MIT](LICENSE).
