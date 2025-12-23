import React from "react";
import Image from "next/image";

interface MediaItem {
  src: string;
  type: "image" | "video";
  alt: string;
  clipId:
    | "clip-goey1"
    | "clip-goey2"
    | "clip-goey3"
    | "clip-goey4"
    | "clip-goey5"
    | "clip-goey6";
  figureClassName?: string;
}

interface ClippedGoeyGalleryProps
  extends React.ComponentPropsWithoutRef<"section"> {
  mediaItems?: MediaItem[];
}

const defaultFigureClasses: Record<MediaItem["clipId"], string> = {
  "clip-goey1":
    "group p-8 hover:p-4 transition-all duration-200 bg-gradient-to-b to-[#bd8122] from-[#c9ad66] rounded-xl",
  "clip-goey2":
    "group p-8 hover:p-4 transition-all duration-200 bg-gradient-to-t to-[#8c8a57] from-[#6f7957] rounded-xl",
  "clip-goey3":
    "group p-8 hover:p-4 transition-all duration-200 bg-gradient-to-t to-[#ba4344] from-[#d5685a] rounded-xl",
  "clip-goey4":
    "group p-8 hover:p-4 transition-all duration-200 bg-gradient-to-t to-[#9b2a08] from-[#f7d578] rounded-xl",
  "clip-goey5":
    "group p-8 hover:p-4 transition-all duration-200 bg-gradient-to-b to-[#022641] from-[#356778] rounded-xl",
  "clip-goey6":
    "group p-8 hover:p-4 transition-all duration-200 bg-gradient-to-b to-[#324644] from-[#89a390] rounded-xl",
};

const ClippedGoeyGallery = React.forwardRef<
  HTMLElement,
  ClippedGoeyGalleryProps
>(({ mediaItems, className, ...props }, ref) => {
  const defaultMediaItems: MediaItem[] = [
    {
      src: "https://images.unsplash.com/photo-1713727660632-841a92340a71?q=80&w=1985&auto=format&fit=crop",
      alt: "Golden abstract art",
      clipId: "clip-goey1",
      type: "image",
    },
    {
      src: "https://images.unsplash.com/photo-1691669151338-a983e71abc05?q=80&w=1978&auto=format&fit=crop",
      alt: "Green abstract art",
      clipId: "clip-goey2",
      type: "image",
      figureClassName: defaultFigureClasses["clip-goey2"],
    },
    {
      src: "https://images.unsplash.com/photo-1683223585116-23e136669e80?q=80&w=1959&auto=format&fit=crop",
      alt: "Red abstract art",
      clipId: "clip-goey3",
      type: "image",
      figureClassName: defaultFigureClasses["clip-goey3"],
    },
    {
      src: "https://images.unsplash.com/photo-1575995864268-5dec34a5bb99?w=500&auto=format&fit=crop",
      alt: "Orange abstract art",
      clipId: "clip-goey4",
      type: "image",
      figureClassName: defaultFigureClasses["clip-goey4"],
    },
    {
      src: "https://images.unsplash.com/photo-1681238093193-1b3a8fcdf225?q=80&w=1962&auto=format&fit=crop",
      alt: "Blue abstract art",
      clipId: "clip-goey5",
      type: "image",
      figureClassName: defaultFigureClasses["clip-goey5"],
    },
    {
      src: "https://images.unsplash.com/photo-1693162640799-9f192c4e0da3?q=80&w=1959&auto=format&fit=crop",
      alt: "Grey abstract art",
      clipId: "clip-goey6",
      type: "image",
      figureClassName: defaultFigureClasses["clip-goey6"],
    },
  ];

  const itemsToRender = mediaItems || defaultMediaItems;

  return (
    <>
      <section
        ref={ref}
        className={`grid grid-cols-3 gap-8 dark:bg-black bg-white border rounded-lg p-5 ${
          className || ""
        }`}
        {...props}
      >
        {itemsToRender.map((item, index) => (
          <figure
            key={index}
            className={
              item.figureClassName || defaultFigureClasses[item.clipId]
            }
          >
            <div style={{ clipPath: `url(#${item.clipId})` }}>
              {item.type === "image" ? (
                <Image
                  src={item.src}
                  alt={item.alt}
                  fill={false}
                  width={500}
                  height={500}
                  className="transition-all duration-300 align-bottom object-cover aspect-square group-hover:scale-110 w-full"
                  style={{ objectFit: "cover" }}
                />
              ) : (
                <video
                  autoPlay
                  muted
                  loop
                  className="transition-all duration-300 align-bottom object-cover aspect-square group-hover:scale-110 w-full"
                >
                  <source src={item.src} type="video/mp4" />
                </video>
              )}
            </div>
          </figure>
        ))}
      </section>
    </>
  );
});

ClippedGoeyGallery.displayName = "ClippedGoeyGallery";

export default ClippedGoeyGallery;
