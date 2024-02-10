# PYTHON-PROJECT-TEMPLATE
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
<!-- [![All Contributors](https://img.shields.io/badge/all_contributors-6-orange.svg?style=flat-square)](#contributors-) -->
<!-- ALL-CONTRIBUTORS-BADGE:END -->

El proyecto "python-project-template" es una plantilla de proyecto de Python que proporciona una estructura y configuración inicial para comenzar a desarrollar aplicaciones en Python de manera organizada y eficiente. Incluye componentes clave como documentación, código fuente, pruebas, configuraciones de GitHub, archivos de Docker, licencia, entre otros. Esta plantilla facilita la colaboración y el mantenimiento a lo largo del tiempo al proporcionar una base sólida y consistente para nuevos proyectos.

```
python-project-template/
│
├── .github/                 # Contiene GitHub workflows y templates para issues y pull requests
│   ├── workflows/           # CI/CD usando GitHub Actions
│   ├── ISSUE_TEMPLATE/      # Plantillas para la creación de issues
│   └── PULL_REQUEST_TEMPLATE.md # Plantilla para la creación de pull requests
│
├── docs/                    # Documentación del proyecto
│   ├── build/               # Archivos generados por Sphinx o otra herramienta de documentación
│   └── source/              # Archivos fuente para la generación de la documentación
│
├── src/                     # Código fuente del proyecto
│   └── package_name/        # Paquete Python (reemplazar con el nombre real del paquete)
│       ├── __init__.py
│       ├── module1.py
│       └── module2.py
│
├── tests/                   # Test suite para el proyecto
│   ├── __init__.py
│   ├── test_module1.py
│   └── test_module2.py
│
├── .gitignore               # Especifica archivos no rastreados por Git
├── .dockerignore            # Especifica archivos no copiados durante la construcción de Docker
├── Dockerfile               # Instrucciones para crear un Docker container para el proyecto
├── LICENSE                  # Licencia del proyecto
├── README.md                # Descripción del proyecto, instrucciones de instalación y uso
├── requirements.txt         # Dependencias del proyecto para producción
├── requirements-dev.txt     # Dependencias del proyecto para desarrollo y pruebas
├── setup.py                 # Script de instalación del paquete
├── pyproject.toml           # Configuración de herramientas de construcción de paquetes, como PEP 518
└── setup.cfg                # Configuraciones para linters y otras herramientas
```

# Descripción de Componentes Clave

- docs/: Documentación del proyecto, posiblemente utilizando herramientas como Sphinx.
- src/: Todo el código fuente del paquete Python.
- tests/: Tests para el proyecto, estructurados con una herramienta de testing como pytest.
- .github/: Configuraciones y plantillas para GitHub, incluyendo CI/CD con GitHub Actions.
- .gitignore: Lista de archivos y carpetas que Git debe ignorar.
- .dockerignore: Lista de archivos y carpetas que Docker debe ignorar al construir imágenes.
- Dockerfile: Configuración para construir un contenedor Docker para el proyecto.
- LICENSE: Licencia bajo la cual se distribuye el proyecto.
- README.md: Información sobre el proyecto, cómo instalarlo, cómo contribuir, etc.
- requirements.txt: Lista de dependencias necesarias para el proyecto en producción.
- requirements-dev.txt: Lista de dependencias necesarias para desarrollo y testing.
- setup.py: Script para instalar el paquete Python usando setuptools.
- pyproject.toml: Configuración para herramientas de construcción de paquetes.
- setup.cfg: Configuraciones para linters (como flake8), formateadores de código (como black), y otras herramientas de desarrollo.

# Instrucciones Adicionales
- Asegúrate de personalizar el README.md con detalles específicos de tu proyecto.
- Actualiza requirements.txt y requirements-dev.txt según las dependencias de tu proyecto.
- Personaliza las plantillas de issue y pull request de GitHub para adaptarlas a las prácticas de tu equipo.
- Configura los workflows de GitHub Actions según tus necesidades de integración continua y despliegue continuo.
- Completa la documentación dentro de docs/ para que otros puedan entender y contribuir fácilmente a tu proyecto.
- Al usar esta plantilla, podrás comenzar nuevos proyectos con una base sólida y consistente, lo que facilitará la colaboración y el mantenimiento a lo largo del tiempo.

## Contributors ✨

Thanks goes to:

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/LuisTellezSirocco"><img src="https://avatars.githubusercontent.com/u/110382845?s=96&v=4" width="100px;" alt=""/><br /><sub><b>Luis Téllez</b></sub></a><br /><a href="https://github.com/LuisTellezSirocco" title="Code">💻</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
