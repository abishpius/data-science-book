import { defineConfig } from "vitepress";

export default defineConfig({
  base: '/data-science-book/',
  title: "The Professional's Introduction to Data Science with Python",
  description:
    "A comprehensive guide for career transitioners into Data Science with Python.",
  themeConfig: {
    logo: "/logo.png", // I'll generate this later
    nav: [
      { text: "Home", link: "/" },
      { text: "Tutorial", link: "/chapters/intro" },
      { 
        text: 'Buy the Book',
        items: [
          { text: 'Kindle eBook ($1.99)', link: 'https://kdp.amazon.com/amazon-dp-action/us/dualbookshelf.marketplacelink/B0G39M1KQM' },
          { text: 'Hardcover ($49.99)', link: 'https://kdp.amazon.com/amazon-dp-action/us/dualbookshelf.marketplacelink/B0G3HV8TBL' }
        ]
      },
      { text: "About Author", link: "https://abishpius.github.io/Abishpius/" },
      { text: "AWS Exam", link: "/aws-exam/index.html" },
    ],
    sidebar: [
      {
        text: "Introduction",
        items: [{ text: "Welcome", link: "/chapters/intro" }],
      },
      {
        text: "The Book",
        items: [
          {
            text: "1. The Data Science Landscape",
            link: "/chapters/chapter-1",
          },
          { text: "2. Python Essentials", link: "/chapters/chapter-2" },
          { text: "3. Mastering Pandas", link: "/chapters/chapter-3" },
          { text: "4. Data Cleaning", link: "/chapters/chapter-4" },
          { text: "5. EDA & Visualization", link: "/chapters/chapter-5" },
          { text: "6. Statistical Foundations", link: "/chapters/chapter-6" },
          { text: "7. Predictive Modeling", link: "/chapters/chapter-7" },
          { text: "8. Classification Algorithms", link: "/chapters/chapter-8" },
          { text: "9. Pattern Discovery", link: "/chapters/chapter-9" },
          { text: "10. The Capstone Project", link: "/chapters/chapter-10" },
        ],
      },
    ],
    socialLinks: [
      { icon: "github", link: "https://github.com/abishpius" },
      { icon: "linkedin", link: "https://www.linkedin.com/in/abishpius/" },
    ],
    footer: {
      message: "Released under the MIT License.",
      copyright: "Copyright © 2025 Abish Pius",
    },
  },
});
