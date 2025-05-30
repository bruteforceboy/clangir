//===- ASTAttrInterfaces.td - CIR AST Interface Definitions -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CIR_INTERFACES_AST_ATTR_INTERFACES
#define MLIR_CIR_INTERFACES_AST_ATTR_INTERFACES

include "mlir/IR/OpBase.td"

let cppNamespace = "::cir" in {
  def ASTDeclInterface : AttrInterface<"ASTDeclInterface"> {
    let methods = [
      InterfaceMethod<"", "bool", "hasOwnerAttr", (ins), [{}],
        /*defaultImplementation=*/ [{
          return $_attr.getAst()->template hasAttr<clang::OwnerAttr>();
        }]
      >,
      InterfaceMethod<"", "bool", "hasPointerAttr", (ins), [{}],
        /*defaultImplementation=*/ [{
          return $_attr.getAst()->template hasAttr<clang::PointerAttr>();
        }]
      >,
      InterfaceMethod<"", "bool", "hasInitPriorityAttr", (ins), [{}],
        /*defaultImplementation=*/ [{
          return $_attr.getAst()->template hasAttr<clang::InitPriorityAttr>();
        }]
      >
    ];
  }

  def ASTNamedDeclInterface : AttrInterface<"ASTNamedDeclInterface",
      [ASTDeclInterface]> {
    let methods = [
      InterfaceMethod<"", "clang::DeclarationName", "getDeclName", (ins), [{}],
        /*defaultImplementation=*/ [{
          return $_attr.getAst()->getDeclName();
        }]
      >,
      InterfaceMethod<"", "llvm::StringRef", "getName", (ins), [{}],
        /*defaultImplementation=*/ [{
          return $_attr.getAst()->getName();
        }]
      >
    ];
  }

  def ASTValueDeclInterface : AttrInterface<"ASTValueDeclInterface",
      [ASTNamedDeclInterface]>;

  def ASTDeclaratorDeclInterface : AttrInterface<"ASTDeclaratorDeclInterface",
      [ASTValueDeclInterface]>;

  def ASTVarDeclInterface : AttrInterface<"ASTVarDeclInterface",
      [ASTDeclaratorDeclInterface]> {
    let methods = [
      InterfaceMethod<"", "void", "mangleDynamicInitializer", (ins "llvm::raw_ostream&":$Out), [{}],
        /*defaultImplementation=*/ [{
          std::unique_ptr<clang::MangleContext> MangleCtx(
            $_attr.getAst()->getASTContext().createMangleContext());
            MangleCtx->mangleDynamicInitializer($_attr.getAst(), Out);
          }]
      >,
      InterfaceMethod<"", "clang::VarDecl::TLSKind", "getTLSKind", (ins), [{}],
        /*defaultImplementation=*/ [{
          return $_attr.getAst()->getTLSKind();
        }]
      >
    ];
  }

  def ASTFunctionDeclInterface : AttrInterface<"ASTFunctionDeclInterface",
      [ASTDeclaratorDeclInterface]> {
    let methods = [
      InterfaceMethod<"", "bool", "isOverloadedOperator", (ins), [{}],
        /*defaultImplementation=*/ [{
          return $_attr.getAst()->isOverloadedOperator();
        }]
      >,
      InterfaceMethod<"", "bool", "isStatic", (ins), [{}],
        /*defaultImplementation=*/ [{
          return $_attr.getAst()->isStatic();
        }]
      >
    ];
  }

  def ASTCXXMethodDeclInterface : AttrInterface<"ASTCXXMethodDeclInterface",
      [ASTFunctionDeclInterface]> {
    let methods = [
      InterfaceMethod<"", "bool", "isCopyAssignmentOperator", (ins), [{}],
        /*defaultImplementation=*/ [{
          if (auto decl = mlir::dyn_cast<clang::CXXMethodDecl>($_attr.getAst()))
            return decl->isCopyAssignmentOperator();
          return false;
        }]
      >,
      InterfaceMethod<"", "bool", "isMoveAssignmentOperator", (ins), [{}],
        /*defaultImplementation=*/ [{
          if (auto decl = mlir::dyn_cast<clang::CXXMethodDecl>($_attr.getAst()))
            return decl->isMoveAssignmentOperator();
          return false;
        }]
      >,
      InterfaceMethod<"", "bool", "isConst", (ins), [{}],
        /*defaultImplementation=*/ [{
          return $_attr.getAst()->isConst();
      }]
      >
    ];
  }

  def ASTCXXConstructorDeclInterface : AttrInterface<"ASTCXXConstructorDeclInterface",
      [ASTCXXMethodDeclInterface]> {
    let methods = [
      InterfaceMethod<"", "bool", "isDefaultConstructor", (ins), [{}],
        /*defaultImplementation=*/ [{
          return $_attr.getAst()->isDefaultConstructor();
        }]
      >,
      InterfaceMethod<"", "bool", "isCopyConstructor", (ins), [{}],
        /*defaultImplementation=*/ [{
          return $_attr.getAst()->isCopyConstructor();
        }]
      >
    ];
  }

  def ASTCXXConversionDeclInterface : AttrInterface<"ASTCXXConversionDeclInterface",
      [ASTCXXMethodDeclInterface]>;

  def ASTCXXDestructorDeclInterface : AttrInterface<"ASTCXXDestructorDeclInterface",
      [ASTCXXMethodDeclInterface]>;

  def ASTTypeDeclInterface : AttrInterface<"ASTTypeDeclInterface",
      [ASTNamedDeclInterface]>;

  def ASTTagDeclInterface : AttrInterface<"ASTTagDeclInterface",
      [ASTTypeDeclInterface]> {
    let methods = [
      InterfaceMethod<"", "clang::TagTypeKind", "getTagKind", (ins), [{}],
        /*defaultImplementation=*/ [{
          return $_attr.getAst()->getTagKind();
        }]
      >
    ];
  }

  def ASTRecordDeclInterface : AttrInterface<"ASTRecordDeclInterface",
      [ASTTagDeclInterface]> {
    let methods = [
      InterfaceMethod<"", "bool", "isLambda", (ins), [{}],
        /*defaultImplementation=*/ [{
          if (auto ast = clang::dyn_cast<clang::CXXRecordDecl>($_attr.getAst()))
            return ast->isLambda();
          return false;
        }]
      >,
      InterfaceMethod<"", "bool", "hasPromiseType", (ins), [{}],
        /*defaultImplementation=*/ [{
          if (!clang::isa<clang::ClassTemplateSpecializationDecl>($_attr.getAst()))
            return false;
          for (const auto *sub : $_attr.getAst()->decls()) {
            if (auto subRec = clang::dyn_cast<clang::CXXRecordDecl>(sub)) {
              if (subRec->getDeclName().isIdentifier() &&
                  subRec->getName() == "promise_type") {
                return true;
              }
            }
          }
          return false;
        }]
      >,
      InterfaceMethod<"", "bool", "isInStdNamespace", (ins), [{}],
        /*defaultImplementation=*/ [{
          return $_attr.getAst()->getDeclContext()->isStdNamespace();
        }]
      >,
      // Note: `getRawDecl` is useful for debugging because it allows dumping
      // the RecordDecl - it should not be used in regular code.
      InterfaceMethod<"", "const clang::RecordDecl *", "getRawDecl", (ins), [{}],
        /*defaultImplementation=*/ [{
          return $_attr.getAst();
        }]
      >
    ];
  }

  def AnyASTFunctionDeclAttr : Attr<
      CPred<"::mlir::isa<::cir::ASTFunctionDeclInterface>($_self)">,
            "AST Function attribute"> {
    let storageType = "::mlir::Attribute";
    let returnType = "::mlir::Attribute";
    let convertFromStorage = "$_self";
    let constBuilderCall = "$0";
  }

  def ASTExprInterface : AttrInterface<"ASTExprInterface"> {}

  def ASTCallExprInterface : AttrInterface<"ASTCallExprInterface",
      [ASTExprInterface]> {
    let methods = [
      InterfaceMethod<"", "bool", "isCalleeInStdNamespace",
                      (ins), [{}], /*defaultImplementation=*/ [{
          // Check that the entity being called is in standard
          // "std" namespace.
          auto callee = $_attr.getAst()->getCallee();
          if (!callee)
            return false;
          auto *ice = mlir::dyn_cast<clang::ImplicitCastExpr>(callee);
          if (!ice)
            return false;

          auto *dre = mlir::dyn_cast_or_null<clang::DeclRefExpr>(ice->getSubExpr());
          if (!dre)
            return false;
          auto qual = dre->getQualifier();
          if (!qual)
            return false;

          // FIXME: should we check NamespaceAlias as well?
          auto nqual = qual->getAsNamespace();
          if (!nqual || !nqual->getIdentifier() ||
              nqual->getName().compare("std") != 0)
            return false;

          return true;
        }]
      >,
      InterfaceMethod<"", "bool", "isStdFunctionCall",
                      (ins "llvm::StringRef":$fn),
                      [{}], /*defaultImplementation=*/ [{
          if (!isCalleeInStdNamespace())
            return false;
          auto fnDecl = $_attr.getAst()->getDirectCallee();
          if (!fnDecl)
            return false;
          // We're looking for `std::<name>`.
          if (!fnDecl->getIdentifier() ||
              fnDecl->getName().compare(fn) != 0)
            return false;
          return true;
        }]
      >,
      InterfaceMethod<"", "bool", "isMemberCallTo",
                      (ins "llvm::StringRef":$fn),
                      [{}], /*defaultImplementation=*/ [{
          auto memberCall = mlir::dyn_cast<clang::CXXMemberCallExpr>($_attr.getAst());
          if (!memberCall)
            return false;
          auto methodDecl = memberCall->getMethodDecl();
          if (!methodDecl)
            return false;
          if (!methodDecl->getIdentifier() ||
              methodDecl->getName().compare(fn) != 0)
            return false;
          return true;
        }]
      >,
      InterfaceMethod<"", "bool", "isIteratorBeginCall",
                      (ins),
                      [{}], /*defaultImplementation=*/ [{
          return isMemberCallTo("begin");
        }]
      >,
      InterfaceMethod<"", "bool", "isIteratorEndCall",
                      (ins),
                      [{}], /*defaultImplementation=*/ [{
          return isMemberCallTo("end");
        }]
      >
    ];

  }


} // namespace cir

#endif // MLIR_CIR_INTERFACES_AST_ATTR_INTERFACES
