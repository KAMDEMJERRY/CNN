// Commencez avec libpqxx + JSON
class ModelRepository {
    pqxx::connection conn_;
    
public:
    ModelRepository() : conn_("postgresql://user:pass@localhost/cnn_models") {}
    
    void saveModel(const CNNModel& model) {
        // Sérialisation manuelle en JSON
        // INSERT simple
    }
    
    CNNModel loadModel(const std::string& name) {
        // SELECT + désérialisation
    }
};