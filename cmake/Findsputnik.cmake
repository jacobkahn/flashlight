# - Try to find Sputnik
#
# The following variables are optionally searched for defaults
#  sputnik_ROOT_DIR:            Base directory where all sputnik components are found
#
# The following are set after configuration is done:
#  sputnik_FOUND
#  sputnik_INCLUDE_DIR
#  sputnik_LIBRARIES

include(FindPackageHandleStandardArgs)

set(sputnik_ROOT_DIR ENV{sputnik_ROOT_DIR} CACHE PATH "Folder contains Sputnik")

find_path(sputnik_INCLUDE_DIR sputnik/sputnik.h
  PATHS ${sputnik_ROOT_DIR}
  PATH_SUFFIXES include
)

find_library(sputnik_LIBRARIES sputnik
  PATHS ${sputnik_ROOT_DIR}
  PATH_SUFFIXES lib lib64)

mark_as_advanced(sputnik_ROOT_DIR sputnik_LIBRARIES sputnik_INCLUDE_DIR)
find_package_handle_standard_args(sputnik DEFAULT_MSG sputnik_INCLUDE_DIR sputnik_LIBRARIES)

add_library(sputnik::sputnik SHARED IMPORTED)
set_target_properties(sputnik::sputnik PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${sputnik_INCLUDE_DIR}"
  IMPORTED_LOCATION "${sputnik_LIBRARIES}"
  )

if(sputnik_FOUND)
  message(STATUS "Found sputnik (include: ${sputnik_INCLUDE_DIR}, library: ${sputnik_LIBRARIES})")
endif()

