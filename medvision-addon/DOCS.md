# MedVision Gesture Engine

Reconnaît des gestes de la main devant une caméra et les transmet à Home Assistant.
Toute l'analyse se fait sur votre machine : aucune image n'est enregistrée ni envoyée.

## Prérequis

- Home Assistant OS ou Supervised, sur une machine 64 bits (Raspberry Pi 5 conseillé, ou mini-PC).
- L'add-on **Mosquitto broker** et l'intégration **MQTT** activés.
- Une webcam USB ou une caméra IP proposant un flux RTSP.

## 1. Enregistrer vos gestes (sur un ordinateur)

La v0.1 n'a pas encore d'interface d'enregistrement dans Home Assistant. Enregistrez les gestes
sur un PC avec la webcam, puis copiez-les :

```bash
uv run medvision record lumiere --label "Allumer la lumière" --preview
```

Copiez ensuite les fichiers `gestures/*.json` dans le dossier `share/medvision/gestures/`
de Home Assistant (avec l'add-on Samba ou File editor), puis appuyez sur le bouton
**Recharger les gestes** de l'appareil MedVision.

## 2. Options

| Option | Description |
|---|---|
| `camera_source` | `0` pour la première webcam USB, ou une URL `rtsp://…` |
| `fps` | Images analysées par seconde (15 suffit) |
| `device_id` / `device_name` | Identifiant et nom de la caméra dans Home Assistant |
| `attention_gesture` | Geste à faire avant toute commande (fortement conseillé) |
| `armed_timeout` | Secondes laissées pour faire la commande après le geste d'attention |
| `motion_gating` | N'active la détection des mains que s'il y a du mouvement (économise le CPU) |
| `mqtt_host` | `auto` pour utiliser Mosquitto de Home Assistant |

## 3. Automatisations

L'appareil crée une entité `event.medvision_<device_id>_geste`. Exemple :
[automations.yaml](https://github.com/VOTRE_COMPTE/medvision/blob/main/examples/automations.yaml)
