add_clang_library(MLIRCIRInterfaces
  ASTAttrInterfaces.cpp
  CIROpInterfaces.cpp
  CIRLoopOpInterface.cpp
  CIRTypeInterfaces.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Interfaces

  DEPENDS
  MLIRCIRASTAttrInterfacesIncGen
  MLIRCIREnumsGen
  MLIRCIRTypeInterfacesIncGen
  MLIRCIRLoopOpInterfaceIncGen
  MLIRCIROpInterfacesIncGen

  LINK_LIBS
  ${dialect_libs}
  MLIRIR
  MLIRSupport
 )
