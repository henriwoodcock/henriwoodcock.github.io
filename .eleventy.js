module.exports = function(eleventyConfig) {
  eleventyConfig.addPassthroughCopy("css/css.css");
  eleventyConfig.addPassthroughCopy("cv/henriwoodcock.pdf");
  eleventyConfig.addPassthroughCopy("CNAME");
};
