const fs = require('fs');

const FILE_PATH = 'apps/nextjs/src/app/api/v1/_routes/dataset/meta/dataset-meta.route.ts';
let code = fs.readFileSync(FILE_PATH, 'utf8');

const targetA = `    // ADDED: Status transition validation
    if (body.status) {
      const allowedStatuses = [
        "raw",
        "latest",
        "preprocessed",
        "validated",
        "archived",
      ];

      if (!allowedStatuses.includes(body.status)) {
        return c.json({ message: \`Invalid status: \${body.status}\` }, 400);
      }

      // Get current dataset to check current status
      let currentDataset;
      if (mongoose.Types.ObjectId.isValid(idOrSlug)) {
        currentDataset = await DatasetMeta.findById(idOrSlug).lean();
      } else {
        currentDataset = await DatasetMeta.findOne({
          collectionName: idOrSlug,
        }).lean();
      }

      if (!currentDataset) {
        return c.json({ message: "Dataset not found" }, 404);
      }`;

const replacementA = `    let currentDataset;
    if (mongoose.Types.ObjectId.isValid(idOrSlug)) {
      currentDataset = await DatasetMeta.findById(idOrSlug).lean();
    } else {
      currentDataset = await DatasetMeta.findOne({
        collectionName: idOrSlug,
      }).lean();
    }

    if (!currentDataset) {
      return c.json({ message: "Dataset not found" }, 404);
    }

    // ADDED: Status transition validation
    if (body.status) {
      const allowedStatuses = [
        "raw",
        "latest",
        "preprocessed",
        "validated",
        "archived",
      ];

      if (!allowedStatuses.includes(body.status)) {
        return c.json({ message: \`Invalid status: \${body.status}\` }, 400);
      }`;
      
code = code.replace(targetA, replacementA);

const targetB = `      console.log(
        \`Status transition validated: \${currentStatus} → \${newStatus}\`,
      );
    }

    let updatedDataset;`;
    
const replacementB = `      console.log(
        \`Status transition validated: \${currentStatus} → \${newStatus}\`,
      );
    }

    const oldCollectionName = (currentDataset as any).collectionName;

    // Handle renaming dataset logic
    if (body.name && body.name.trim() !== (currentDataset as any).name) {
      const newName = body.name.trim();
      body.name = newName;
      
      const newCollectionName = newName.replace(/[^a-zA-Z0-9\\s]/g, "").replace(/\\s+/g, " ");
      body.collectionName = newCollectionName;
      
      const isAPI = (currentDataset as any).isAPI;
      const fileExt = (currentDataset as any).fileType || "json";
      body.filename = isAPI ? \`Dataset \${newName}.\${fileExt}\` : \`\${newName}.\${fileExt}\`;
    } else if (body.collectionName && body.collectionName !== oldCollectionName) {
      body.collectionName = body.collectionName.trim().replace(/[^a-zA-Z0-9\\s]/g, "").replace(/\\s+/g, " ");
    }

    if (body.collectionName && body.collectionName !== oldCollectionName) {
      const newCollectionName = body.collectionName;
      const existingMeta = await DatasetMeta.findOne({
        collectionName: newCollectionName,
        _id: { $ne: (currentDataset as any)._id }
      }).lean();

      if (existingMeta) {
        return c.json({ message: "Nama koleksi ini sudah digunakan oleh dataset lain" }, 400);
      }

      const mongoDb = mongoose.connection.db;
      if (mongoDb) {
        try {
          await mongoDb.collection(oldCollectionName).rename(newCollectionName);
          console.log(\`[RENAME] collection \${oldCollectionName} to \${newCollectionName}\`);
          if (mongoose.models[oldCollectionName]) delete mongoose.models[oldCollectionName];
        } catch (e: any) {
          if (!e.message.includes('not found') && !e.message.includes('ns not found')) {
            console.error("[RENAME ERROR]", e);
          }
        }

        try {
          await mongoDb.collection(\`\${oldCollectionName}_cleaned\`).rename(\`\${newCollectionName}_cleaned\`);
          console.log(\`[RENAME] collection \${oldCollectionName}_cleaned to \${newCollectionName}_cleaned\`);
          if (mongoose.models[\`\${oldCollectionName}_cleaned\`]) delete mongoose.models[\`\${oldCollectionName}_cleaned\`];
        } catch (e: any) {}

        try {
          if (mongoose.models['PreprocessingReport']) {
            await mongoose.model('PreprocessingReport').updateMany(
              { original_collection_name: oldCollectionName },
              { 
                $set: { 
                  original_collection_name: newCollectionName,
                  cleaned_collection_name: \`\${newCollectionName}_cleaned\`
                } 
              }
            );
          }
        } catch(e) { console.error("[RENAME ERROR] Reports update:", e); }
      }
    }

    let updatedDataset;`;
    
code = code.replace(targetB, replacementB);

fs.writeFileSync(FILE_PATH, code);
console.log("Patched successfully!");
