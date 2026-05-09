require("dotenv").config();
const { Sequelize, DataTypes, Op } = require("sequelize");

// Create a Sequelize instance using environment variables
const sequelize = new Sequelize(
  process.env.DB_DATABASE,
  process.env.DB_USERNAME,
  process.env.DB_PASSWORD,
  {
    host: process.env.DB_HOST,

    dialect: process.env.DB_DIALECT,
    logging: false,
  },
);

const sequelize2 = new Sequelize(
  process.env.DB2_DATABASE,
  process.env.DB2_USERNAME,
  process.env.DB2_PASSWORD,
  {
    host: process.env.DB2_HOST,
    dialect: process.env.DB2_DIALECT,
    logging: false,
    pool: {
      max: 5,
      min: 0,
      acquire: 300000,
      idle: 10000,
    },
    dialectOptions: {
      options: {
        requestTimeout: 300000, // Timeout for queries (in ms)
      },
    },
  },
);

// Test the connection
(async () => {
  try {
    await sequelize.authenticate();
    console.log(
      "Connection to the database CRM has been established successfully.",
    );
  } catch (error) {
    console.error("Unable to connect to CRM:", error);
  }

  try {
    await sequelize2.authenticate();
    console.log(
      "Connection to the database UNIGES has been established successfully.",
    );
  } catch (error) {
    console.error("Unable to connect to UNIGES");
  }
})();

// Import models
const UserModel = require("./models/users")(sequelize, DataTypes);
const VisiteProductsModel = require("./models/visite_products")(
  sequelize,
  DataTypes,
);
const VisiteModel = require("./models/visite")(sequelize, DataTypes);
const ProspectModel = require("./models/prospect")(sequelize, DataTypes);
const SpecialteModel = require("./models/specialite")(sequelize, DataTypes);
const DelegationModel = require("./models/delegation")(sequelize, DataTypes);
const GouverneratModel = require("./models/gouvernerat")(sequelize, DataTypes);
const ActivitetModel = require("./models/activite")(sequelize, DataTypes);
const PotentieltModel = require("./models/potentiel")(sequelize, DataTypes);
const ProductsModel = require("./models/products")(sequelize, DataTypes);
const VisiteTypeModel = require("./models/type_visite")(sequelize, DataTypes);
const AffectationModel = require("./models/affectation")(sequelize, DataTypes);
const PrdSpecModel = require("./models/prdspec")(sequelize, DataTypes);
const User_TypeModel = require("./models/user_type")(sequelize, DataTypes);
const Date_jouvModel = require("./models/date_jouv")(sequelize, DataTypes);
const PointageModel = require("./models/pointage")(sequelize, DataTypes);
const DemandeModel = require("./models/demande")(sequelize, DataTypes);
const TypeDemandeModel = require("./models/type_demande")(sequelize, DataTypes);
const Commentaire = require("./models/commentaire")(sequelize, DataTypes);
const BaModel = require("./models/ba_demandes")(sequelize, DataTypes);
const TypePayModel = require("./models/type_pay")(sequelize, DataTypes);
const ListeModel = require("./models/liste")(sequelize, DataTypes);
const ProgVisiteModel = require("./models/prog_visite")(sequelize, DataTypes);
const RatioGroModel = require("./models/ratiogro")(sequelize, DataTypes);
const ObjectifGroModel = require("./models/obj_gro_bu")(sequelize, DataTypes);
const TotVente = require("./models/ca_tot_vente")(sequelize, DataTypes);
const obj_gro_bu = require("./models/obj_gro_bu")(sequelize, DataTypes);
const ProdZone = require("./models/products_zone")(sequelize, DataTypes);
const DateJouvModel = require("./models/date_jouv")(sequelize, DataTypes);
const ProgToursModel = require("./models/prog_tourne")(sequelize, DataTypes);
const TypePointage = require("./models/type_conger")(sequelize, DataTypes);
const NoteFraisModel = require("./models/note_frais")(sequelize, DataTypes);
const RaportModel = require("./models/rapport_dm")(sequelize, DataTypes);
const UsersZoneModel = require("./models/user_bu")(sequelize, DataTypes);
const GpsModel = require("./models/gps_stops")(sequelize, DataTypes);
const AireModel = require("./models/airet")(sequelize, DataTypes);

const AffectationPhModel = require("./models/affectation_uniges_ph")(
  sequelize,
  DataTypes,
);
const DcrepCodeModel = require("./models/dcrepcode")(sequelize, DataTypes);
const EtablisementModel = require("./models/etablissement")(
  sequelize,
  DataTypes,
);
const PostalCodeModel = require("./models/postal_code")(sequelize, DataTypes);

const Grm_budget_annuelModel = require("./models/grm_budget_annuel_zone")(
  sequelize,
  DataTypes,
);
const ZoneModel = require("./models/zone")(sequelize, DataTypes);
const Grm_budget_annuel_zoneModel = require("./models/grm_budget_annuel_zone")(
  sequelize,
  DataTypes,
);
const CommentaireModel = require("./models/commentaire")(sequelize, DataTypes);
const Prod_categorieModel = require("./models/prod_categorie")(
  sequelize,
  DataTypes,
);
const ProspectsDemandesModels = require("./models/prospects_demandes")(
  sequelize,
  DataTypes,
);
const ProdObjModel = require("./models/productsobj")(sequelize, DataTypes);
const models = {
  Users: UserModel,
  Visite: VisiteModel,
  Prospect: ProspectModel,
  Specialite: SpecialteModel,
  Potentiel: PotentieltModel,
  Activite: ActivitetModel,
  Gouvernerat: GouverneratModel,
  Delegation: DelegationModel,
  VisiteProducts: VisiteProductsModel,
  TypeVisite: VisiteTypeModel,
  Affectation: AffectationModel,
  Prdspec: PrdSpecModel,
  User_Type: User_TypeModel,
  Date_jouv: Date_jouvModel,
  Pointage: PointageModel,
  Demande: DemandeModel,
  TypeDemande: TypeDemandeModel,
  Grm_budget_annuel: Grm_budget_annuelModel,
  Commentaire: CommentaireModel,
  ProspectsDemande: ProspectsDemandesModels,
  BA: BaModel,
  TypePay: TypePayModel,
  Grm_budget_annuel_zone: Grm_budget_annuel_zoneModel,
  Liste: ListeModel,
  ProgVsite: ProgVisiteModel,
  RatioGro: RatioGroModel,
  ObjectifGroBu: ObjectifGroModel,
  Products: ProductsModel,
  ProdGamme: Prod_categorieModel,
  ProdObj: ProdObjModel,
  Zone: ZoneModel,
  TotVente: TotVente,
  ObjGroBu: obj_gro_bu,
  ProdZone: ProdZone,
  DateJouv: DateJouvModel,
  ProgTours: ProgToursModel,
  TypePointage: TypePointage,
  NoteFrais: NoteFraisModel,
  Rapport: RaportModel,
  UsersZone: UsersZoneModel,
  AffectationPh: AffectationPhModel,
  DcrepCode: DcrepCodeModel,
  Etablisement: EtablisementModel,
  PostalCode: PostalCodeModel,
  GpsStops: GpsModel,
  Aire: AireModel,
};

/*ProductsModel.associate = function (models) {
 
 
};
ProdObjModel.associate = function (models) {
  ProdObjModel.belongsTo(models.Products, {
    foreignKey: "idPrd",
  });
  
};*/
ProgVisiteModel.associate = function (models) {
  // Relation : Une visite programmée appartient à un utilisateur
  ProgVisiteModel.belongsTo(models.Users, {
    foreignKey: "user_id",
  });
  // Relation : Une visite programmée concerne un prospect
  ProgVisiteModel.belongsTo(models.Prospect, {
    foreignKey: "pros_id",
  });
};
ProspectModel.associate = function (models) {
  // Relation : Un prospect a une spécialité
  ProspectModel.belongsTo(models.Specialite, {
    foreignKey: "spec",
    as: "specialite",
  });

  // Relation : Un prospect peut avoir plusieurs objectifs GRO BU
  ProspectModel.hasMany(models.ObjectifGroBu, {
    foreignKey: "id_gro",
    as: "obj_gro_bu",
  });

  // Relation : Un prospect est lié à un ratio GRO
  ProspectModel.belongsTo(models.RatioGro, {
    foreignKey: "id",
  });

  // Relation : Un prospect appartient à une délégation
  ProspectModel.belongsTo(models.Delegation, {
    foreignKey: "delegation",
    as: "delegation_name",
  });

  // Relation : Un prospect est créé par un utilisateur
  ProspectModel.belongsTo(models.Users, {
    as: "user_creator",
    foreignKey: "cree_par",
  });

  // Relation : Un prospect appartient à un gouvernorat
  ProspectModel.belongsTo(models.Gouvernerat, {
    foreignKey: "gouvernorat",
    as: "gouvernerat_name",
  });

  // Relation : Un prospect a une activité
  ProspectModel.belongsTo(models.Activite, {
    foreignKey: "activite",
    as: "activite_name",
  });

  // Relation : Un prospect est lié à un établissement (optionnel)
  ProspectModel.belongsTo(models.Etablisement, {
    allowNull: true,
    foreignKey: "etablissement",
    as: "etablissement_name",
  });

  // Relation : Un prospect a un potentiel
  ProspectModel.belongsTo(models.Potentiel, {
    foreignKey: "potentiel",
    as: "potentiel_name",
  });

  // Relation : Un prospect peut avoir plusieurs visites
  ProspectModel.hasMany(models.Visite, {
    foreignKey: "id_pros",
    as: "prospect",
  });

  // Relation : Un prospect est affecté à plusieurs utilisateurs (relation N:N via Affectation)
  ProspectModel.belongsToMany(models.Users, {
    through: models.Affectation,
    foreignKey: "id_prospect",
    otherKey: "id_deleg",
    as: "users", // Use this alias to access visits associated with a product
  });
};

UserModel.associate = function (models) {
  // Relation : Un utilisateur peut effectuer plusieurs visites
  UserModel.hasMany(models.Visite, {
    foreignKey: "id_visiteur",
    as: "users",
  });

  // Relation : Un utilisateur peut être affecté à plusieurs prospects (relation N:N via Affectation)
  UserModel.belongsToMany(models.Prospect, {
    through: models.Affectation,
    foreignKey: "id_deleg", // Clé étrangère dans la table Affectation pour l'Utilisateur
    otherKey: "id_prospect", // Clé étrangère dans la table Affectation pour le Prospect
    as: "prospects", // Alias pour accéder aux prospects associés
  });

  // Relation : Un utilisateur a un type (rôle)
  UserModel.belongsTo(models.User_Type, {
    foreignKey: "type",
  });
};
DemandeModel.associate = function (models) {
  // Relation : Une demande a un type
  DemandeModel.belongsTo(models.TypeDemande, {
    foreignKey: "type",
  });
  // Relation : Une demande est faite par un utilisateur
  DemandeModel.belongsTo(models.Users, {
    foreignKey: "id_user",
  });
};
BaModel.associate = function (models) {
  // Relation : Un bon d'achat est créé par un utilisateur
  BaModel.belongsTo(models.Users, {
    foreignKey: "cree_par",
    as: "user_create",
  });
  // Relation : Un bon d'achat a un type de paiement
  BaModel.belongsTo(models.TypePay, {
    foreignKey: "type_pay",
    as: "modeP",
  });
  // Relation : Un bon d'achat est validé par un utilisateur
  BaModel.belongsTo(models.Users, {
    foreignKey: "valider_par",
    as: "user_validate",
  });
  // Relation : Un bon d'achat est lié à une demande
  BaModel.belongsTo(models.Demande, {
    foreignKey: "demande_id",
  });
};

CommentaireModel.associate = function (models) {
  // Relation : Un commentaire est créé par un utilisateur
  CommentaireModel.belongsTo(models.Users, {
    foreignKey: "cree_par",
  });
};
ProspectsDemandesModels.associate = function (models) {
  // Relation : Une demande prospect est liée à un prospect
  ProspectsDemandesModels.belongsTo(models.Prospect, {
    foreignKey: "id_prospect",
  });
};
PrdSpecModel.associate = function (models) {
  // Relation : Une spécification produit est liée à un produit
  PrdSpecModel.belongsTo(models.Products, {
    foreignKey: "prd",
  });
  // Relation : Une spécification produit est liée à une zone
  PrdSpecModel.belongsTo(models.Zone, {
    foreignKey: "idr",
  });
};
/*UserModel.associate = function (models) {
  UserModel.hasMany(models.User_Bu, {
    foreignKey: "user",
    as: "user_bu",
  });

}*/

AffectationModel.associate = function (models) {
  // Make sure to reference models using models.YourModelName
  // Relation : Une affectation concerne un prospect
  AffectationModel.belongsTo(models.Prospect, {
    foreignKey: "id_prospect",
  });
  // Relation : Une affectation concerne un utilisateur
  AffectationModel.belongsTo(models.Users, {
    foreignKey: "id_deleg",
  });
};

VisiteModel.associate = function (models) {
  // Relation : Une visite concerne un prospect
  VisiteModel.belongsTo(models.Prospect, {
    foreignKey: "id_pros",
    as: "prospect",
  });
  // Relation : Une visite a un type
  VisiteModel.belongsTo(models.TypeVisite, {
    foreignKey: "type",
    as: "type_visite",
  });

  // Relation : Une visite est effectuée par un utilisateur
  VisiteModel.belongsTo(models.Users, {
    foreignKey: "id_visiteur",
    as: "users",
  });

  // Relation : Une visite peut concerner plusieurs produits (relation N:N via VisiteProducts)
  VisiteModel.belongsToMany(models.Products, {
    through: models.VisiteProducts,
    foreignKey: "id_visite",
    otherKey: "produits",
    as: "products", // Use this alias for retrieving products related to a visite
  });
};

ProductsModel.associate = function (models) {
  // Relation : Un produit peut être associé à plusieurs visites (relation N:N via VisiteProducts)
  ProductsModel.belongsToMany(models.Visite, {
    through: models.VisiteProducts,
    foreignKey: "produits",
    otherKey: "id_visite",
    as: "visites", // Use this alias to access visits associated with a product
  });
  // Relation : Un produit appartient à une gamme
  ProductsModel.belongsTo(models.ProdGamme, {
    foreignKey: "gamme_id",
  });
  // Relation : Un produit appartient à une aire
  ProductsModel.belongsTo(models.Aire, {
    foreignKey: "aire",
    as: "airet",
  });
  // Relation : Un produit peut avoir plusieurs objectifs
  ProductsModel.hasMany(models.ProdObj, {
    foreignKey: "idPrd",
  });
  // Relation : Un produit peut être lié à plusieurs zones
  ProductsModel.hasMany(models.ProdZone, {
    foreignKey: "id_prd",
  });
};

VisiteProductsModel.associate = function (models) {
  // Relation : Un détail de visite est lié à un produit
  VisiteProductsModel.belongsTo(models.Products, {
    foreignKey: "produits",
  });
};

ListeModel.associate = function (models) {
  // Associate user_id with the Users model
  // Relation : Une liste appartient à un utilisateur
  ListeModel.belongsTo(models.Users, {
    foreignKey: "user_id",
    as: "user", // Alias for the user associated with this list
  });

  // Associate supID with the Users model
  // Relation : Une liste a un superviseur
  ListeModel.belongsTo(models.Users, {
    foreignKey: "supID",
    as: "supervisor", // Alias for the supervisor
  });
};
// Call the associations
models.Prospect.associate(models);
models.Users.associate(models);
models.Visite.associate(models);
models.Products.associate(models);
//models.ProdObj.associate(models);
models.Affectation.associate(models);
models.Prdspec.associate(models);
models.VisiteProducts.associate(models);
models.Demande.associate(models);
models.ProspectsDemande.associate(models);
models.Commentaire.associate(models);
models.BA.associate(models);
models.Liste.associate(models);
models.ProgVsite.associate(models);

// Export the sequelize instance and models
module.exports = {
  sequelize,
  sequelize2,
  models,
};
