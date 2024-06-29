// Learn more https://docs.expo.io/guides/customizing-metro
const { getDefaultConfig } = require("expo/metro-config");

// module.exports = getDefaultConfig(__dirname);

module.exports = (() => {
  const config = getDefaultConfig(__dirname);
  config.resolver.assetExts.push("bin");
  return config;
})();
