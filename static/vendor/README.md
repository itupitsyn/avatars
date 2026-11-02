Копии библиотек с jsdelivr, чтобы страница не зависела от CDN: рендер кружочков (video/note.py) иногда
подвисал на первом запросе к нему. Все под MIT. Пути повторяют пакеты — относительные импорты внутри работают.

- three@0.170.0 — build/three.module.js и нужные куски examples/jsm (загрузчики GLTF и FBX с их зависимостями)
- @pixiv/three-vrm@3.5.5, @pixiv/three-vrm-animation@3.5.5 — lib/*.module.min.js
