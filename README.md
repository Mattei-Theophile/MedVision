# MedVision : piloter sa maison par gestes

Une caméra reconnaît des gestes de la main (et, à terme, des signes de la langue des signes)
puis Home Assistant exécute l'action associée : allumer une lumière, ouvrir un volet, prévenir
un proche…

Le projet s'adresse à tout le monde, et en particulier aux personnes âgées, sourdes ou
malentendantes, ou à mobilité réduite.

- **100 % local** : l'analyse se fait sur votre machine, sans cloud.
- **Respect de la vie privée** : aucune image n'est stockée ni transmise. Seuls des points
  de la main (21 coordonnées) sont calculés, et seul le nom du geste reconnu sort du moteur.
- **Vos propres gestes** : 5 exemples suffisent pour apprendre un geste, sans entraînement.
- **Sécurité** : un *geste d'attention* optionnel doit précéder chaque commande, pour éviter
  les déclenchements involontaires.

> **Statut : v0.1 expérimentale.** Elle reconnaît des **gestes personnalisés** et des **signes
> isolés** que vous enregistrez vous-même. Elle ne traduit **pas** des phrases en LSF. Vos
> retours décideront de la suite : voir [Donner son avis](#donner-son-avis).

## Comment ça marche

```
Caméra ──► détection de mouvement ──► MediaPipe (21 points/main) ──► normalisation
   (images jetées aussitôt)       (économise le CPU)                       │
                                                                           ▼
Home Assistant ◄── MQTT (event « lumiere ») ◄── machine à états ◄── comparaison DTW
  automatisations                          (attention, anti-rebond)   aux exemples enregistrés
```

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
```

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
2. Installez **MedVision Gesture Engine**, choisissez la caméra dans les options et démarrez-le.
3. Copiez vos fichiers `gestures/*.json` dans `share/medvision/gestures/` et appuyez sur
   **Recharger les gestes**.

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

### Entités créées automatiquement (découverte MQTT)

| Entité | Rôle |
|---|---|
| `event.medvision_salon_geste` | Déclencheur : un événement par geste (`event_type` = nom du geste) |
| `sensor.medvision_salon_dernier_geste` | Dernier geste reconnu |
| `binary_sensor.medvision_salon_main` | Une main est visible |
| `binary_sensor.medvision_salon_attente` | Geste d'attention reconnu, commande attendue |
| `button.medvision_salon_recharger` | Relit les gestes sans redémarrer |

Voir [examples/automations.yaml](examples/automations.yaml) : lumière, volets, alerte d'un
proche, et confirmation visuelle pour les personnes sourdes.

## Configuration

Toutes les options sont documentées dans [config.example.yaml](config.example.yaml). Chacune
peut aussi passer par une variable d'environnement `MEDVISION_<OPTION>`. Les plus utiles :

| Option | Défaut | Effet |
|---|---|---|
| `camera_source` | `0` | Index de webcam ou URL RTSP (préférez le flux secondaire basse définition) |
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
| Réseau | Idéalement, caméra et serveur sur un VLAN ou un SSID dédié à la domotique |

Le moteur affiche toutes les minutes une ligne `Performance : … img/s, … % analysées`.
Merci de l'indiquer dans vos retours.

## Développement

```bash
uv sync                    # dépendances + outils de dev
uv run pytest              # tests (sans caméra ni MediaPipe : mains synthétiques)
uv run ruff check . && uv run ruff format .
```

```
src/medvision/
  config.py      options (YAML, options.json d'add-on, variables d'environnement)
  features.py    normalisation des 21 points de la main
  classifier.py  bibliothèque de gestes, DTW, calibration automatique des seuils
  engine.py      machine à états : fenêtre glissante, stabilité, geste d'attention
  store.py       gestes au format JSON
  vision.py      caméra, détection de mouvement, MediaPipe Hand Landmarker
  mqtt.py        publication et découverte Home Assistant
  addon.py       point d'entrée de l'add-on
  cli.py         commandes record / list / delete / run / download-model
```

## Feuille de route

- [x] v0.1 : gestes personnalisés à une main, MQTT et découverte HA, add-on, Docker
- [ ] Interface web d'enregistrement des gestes dans Home Assistant (ingress)
- [ ] Gestes à deux mains et posture du corps (détection de chute sans image)
- [ ] Vocabulaire LSF de base, construit **avec** des signeurs et signeuses
- [ ] Portage du moteur sur une caméra embarquée à bas coût (Rockchip RV1106)

## Vie privée et sécurité

- Les images restent en mémoire le temps d'une analyse (environ 1/15 s), puis sont détruites.
- Les fichiers de gestes ne contiennent que des coordonnées normalisées.
- N'installez pas de caméra chez quelqu'un sans son consentement éclairé.
- Ce logiciel n'est **pas** un dispositif médical, ni un système d'alarme certifié. Ne vous
  en servez pas comme unique moyen d'appeler les secours.

## Donner son avis

Ouvrez une [issue « Retour de test »](../../issues/new/choose). Tous les profils
nous intéressent, en particulier les personnes âgées, sourdes ou signeuses, et leurs aidants.

Licence : [MIT](LICENSE).
