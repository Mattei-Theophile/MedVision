# Journal des modifications

## 0.3.0

- **Radar mmWave HLK-LD2450** : nouveau capteur (`sensor_type: ld2450`), sur port série ou
  passerelle réseau (`socket://ip:port`). Présence fiable dans le noir, nombre de personnes,
  distance, et reconnaissance de gestes amples (s'approcher, pas de côté, balayage du bras).
- **Architecture *adapter*** : caméra, radar et capteur simulé partagent une même interface ;
  le moteur, MQTT et l'interface web ne dépendent plus d'un capteur précis.
- **Interface web** dans la barre latérale de Home Assistant (ingress) : état en direct,
  enregistrement des gestes avec compte à rebours visuel, suppression, rechargement.
  Aucune image n'y transite, seulement des points.
- Gestes multi-capteurs : chaque fichier indique le capteur qui l'a produit ; les fichiers de la
  v0.1 restent valides.
- Capteur `simulation` pour tester les automatisations sans matériel.
- Accès aux ports série (`uart: true`).

## 0.1.0

- Première version : gestes personnalisés à une main, MQTT et découverte Home Assistant.
