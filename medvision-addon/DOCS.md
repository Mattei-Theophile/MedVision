# MedVision Gesture Engine

Reconnaît des gestes devant une **caméra** ou un **radar mmWave HLK-LD2450** et les transmet à
Home Assistant. Toute l'analyse se fait sur votre machine : aucune image n'est enregistrée ni
envoyée (le radar ne produit même aucune image).

## Prérequis

- Home Assistant OS ou Supervised, sur une machine 64 bits (Raspberry Pi 5 conseillé, ou mini-PC).
- L'add-on **Mosquitto broker** et l'intégration **MQTT** activés.
- Au choix :
  - une webcam USB ou une caméra IP proposant un flux RTSP ;
  - un radar **HLK-LD2450** relié en USB (adaptateur USB-UART 3,3 V) ou via une passerelle
    réseau (ESP32 avec ESPHome *stream server*, ser2net…).

## 1. Choisir le capteur

| Option | Caméra | Radar LD2450 |
|---|---|---|
| `sensor_type` | `camera` | `ld2450` |
| Source | `camera_source` : `0` ou `rtsp://…` | `radar_port` : `/dev/ttyUSB0` ou `socket://192.168.50.20:6638` |
| Gestes | mains, doigts, signes isolés | gestes amples du corps ou du bras |
| Fonctionne dans le noir | non (sauf caméra IR) | oui |

Branchez le radar, puis repérez son port dans **Paramètres › Système › Matériel** (préférez le
chemin stable `/dev/serial/by-id/…`).

## 2. Enregistrer vos gestes

Ouvrez **MedVision** dans la barre latérale de Home Assistant :

1. saisissez un nom (`lumiere`, `volet_ouvrir`…) et un libellé ;
2. cliquez sur **Commencer l'enregistrement** et suivez le compte à rebours affiché en grand ;
3. répétez le geste 5 fois ; la qualité calculée s'affiche dans la liste.

Enregistrez aussi un geste `repos` (position neutre) : il est reconnu mais jamais envoyé, et
réduit fortement les déclenchements involontaires.

Les gestes sont rangés dans `share/medvision/gestures/` : vous pouvez aussi y copier des
fichiers enregistrés sur un PC avec `medvision record`. Un geste n'est reconnu que par le type
de capteur qui a servi à l'enregistrer.

## 3. Options

| Option | Description |
|---|---|
| `sensor_type` | `camera`, `ld2450` ou `simulation` (rejoue vos gestes, pour tester) |
| `camera_source` | `0` pour la première webcam USB, ou une URL `rtsp://…` |
| `radar_port` | Port série du radar, ou `socket://ip:port` |
| `radar_max_distance` | Distance maximale (m) : les personnes plus loin sont ignorées |
| `fps` | Images analysées par seconde pour la caméra (15 suffit) |
| `device_id` / `device_name` | Identifiant et nom du capteur dans Home Assistant |
| `attention_gesture` | Geste à faire avant toute commande (fortement conseillé) |
| `armed_timeout` | Secondes laissées pour faire la commande après le geste d'attention |
| `motion_gating` | Caméra : n'active la détection des mains que s'il y a du mouvement |
| `mqtt_host` | `auto` pour utiliser Mosquitto de Home Assistant |

L'add-on fait tourner un seul capteur. Pour combiner une caméra et un radar, lancez le second
avec Docker ou `medvision run` (sur la même machine ou une autre), avec un `device_id` différent.

## 4. Entités et automatisations

- `event.medvision_<device_id>_geste` : un événement par geste (`event_type` = nom du geste) ;
- `binary_sensor.medvision_<device_id>_main` (caméra) ou `…_presence` (radar) ;
- radar : `sensor.medvision_<device_id>_targets` (personnes) et `…_distance` (m).

Exemples : [automations.yaml](https://github.com/VOTRE_COMPTE/medvision/blob/main/examples/automations.yaml)
