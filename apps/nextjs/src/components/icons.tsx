import {
  AlertTriangle,
  ArrowRight,
  Check,
  ChevronLeft,
  ChevronRight,
  CircuitBoardIcon,
  Command,
  CreditCard,
  File,
  FileText,
  HelpCircle,
  Image,
  Laptop,
  LayoutDashboardIcon,
  Loader2,
  LogIn,
  LucideIcon,
  LucideProps,
  LucideShoppingBag,
  Moon,
  MoreVertical,
  Pizza,
  Plus,
  Settings,
  SunMedium,
  Trash,
  Twitter,
  User,
  UserCircle2Icon,
  UserPen,
  UserX2Icon,
  X,
} from "lucide-react";

export type Icon = LucideIcon;

export const Icons = {
  dashboard: LayoutDashboardIcon,
  logo: Command,
  login: LogIn,
  close: X,
  product: LucideShoppingBag,
  spinner: Loader2,
  kanban: CircuitBoardIcon,
  chevronLeft: ChevronLeft,
  chevronRight: ChevronRight,
  trash: Trash,
  employee: UserX2Icon,
  post: FileText,
  page: File,
  userPen: UserPen,
  user2: UserCircle2Icon,
  media: Image,
  settings: Settings,
  billing: CreditCard,
  ellipsis: MoreVertical,
  add: Plus,
  warning: AlertTriangle,
  user: User,
  arrowRight: ArrowRight,
  help: HelpCircle,
  pizza: Pizza,
  sun: SunMedium,
  moon: Moon,
  laptop: Laptop,
  gitHub: ({ ...props }: LucideProps) => (
    <svg aria-hidden="true" focusable="false" data-prefix="fab" data-icon="github" role="img" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 496 512" {...props}>
      <path
        fill="currentColor"
        d="M165.9 397.4c0 2-2.3 3.6-5.2 3.6-3.3 .3-5.6-1.3-5.6-3.6 0-2 2.3-3.6 5.2-3.6 3-.3 5.6 1.3 5.6 3.6zm-31.1-4.5c-.7 2 1.3 4.3 4.3 4.9 2.6 1 5.6 0 6.2-2s-1.3-4.3-4.3-5.2c-2.6-.7-5.5 .3-6.2 2.3zm44.2-1.7c-2.9 .7-4.9 2.6-4.6 4.9 .3 2 2.9 3.3 5.9 2.6 2.9-.7 4.9-2.6 4.6-4.6-.3-1.9-3-3.2-5.9-2.9zM244.8 8C106.1 8 0 113.3 0 252c0 110.9 69.8 205.8 169.5 239.2 12.8 2.3 17.3-5.6 17.3-12.1 0-6.2-.3-40.4-.3-61.4 0 0-70 15-84.7-29.8 0 0-11.4-29.1-27.8-36.6 0 0-22.9-15.7 1.6-15.4 0 0 24.9 2 38.6 25.8 21.9 38.6 58.6 27.5 72.9 20.9 2.3-16 8.8-27.1 16-33.7-55.9-6.2-112.3-14.3-112.3-110.5 0-27.5 7.6-41.3 23.6-58.9-2.6-6.5-11.1-33.3 2.6-67.9 20.9-6.5 69 27 69 27 20-5.6 41.5-8.5 62.8-8.5s42.8 2.9 62.8 8.5c0 0 48.1-33.6 69-27 13.7 34.7 5.2 61.4 2.6 67.9 16 17.7 25.8 31.5 25.8 58.9 0 96.5-58.9 104.2-114.8 110.5 9.2 7.9 17 22.9 17 46.4 0 33.7-.3 75.4-.3 83.6 0 6.5 4.6 14.4 17.3 12.1C428.2 457.8 496 362.9 496 252 496 113.3 383.5 8 244.8 8zM97.2 352.9c-1.3 1-1 3.3 .7 5.2 1.6 1.6 3.9 2.3 5.2 1 1.3-1 1-3.3-.7-5.2-1.6-1.6-3.9-2.3-5.2-1zm-10.8-8.1c-.7 1.3 .3 2.9 2.3 3.9 1.6 1 3.6 .7 4.3-.7 .7-1.3-.3-2.9-2.3-3.9-2-.6-3.6-.3-4.3 .7zm32.4 35.6c-1.6 1.3-1 4.3 1.3 6.2 2.3 2.3 5.2 2.6 6.5 1 1.3-1.3 .7-4.3-1.3-6.2-2.2-2.3-5.2-2.6-6.5-1zm-11.4-14.7c-1.6 1-1.6 3.6 0 5.9 1.6 2.3 4.3 3.3 5.6 2.3 1.6-1.3 1.6-3.9 0-6.2-1.4-2.3-4-3.3-5.6-2z"
      ></path>
    </svg>
  ),
  msfile: ({ ...props }: LucideProps) => (
    <svg xmlns="http://www.w3.org/2000/svg" width={64} height={64} fill="none" {...props}>
      <g clipPath="url(#a)">
        <path
          fill="#9D9C9B"
          fillRule="evenodd"
          d="M16 5.587v52.826C16 61.499 18.462 64 21.5 64h37c3.038 0 5.5-2.502 5.5-5.587v-43.71a5.632 5.632 0 0 0-1.632-3.973l-9.072-9.115A5.456 5.456 0 0 0 49.428 0H21.5C18.462 0 16 2.502 16 5.587Zm3.5 52.826V5.587c0-1.122.895-2.031 2-2.031h27.25v12.19c0 2.104 1.679 3.81 3.75 3.81h8v38.857c0 1.122-.895 2.031-2 2.031h-37c-1.105 0-2-.91-2-2.031ZM60.5 16v-1.297c0-.543-.214-1.063-.594-1.445L52.25 5.565v10.181c0 .14.112.254.25.254h8Z"
          clipRule="evenodd"
        />
        <path fill="#6BB3EC" fillRule="evenodd" d="M41 26.25c0-.966.783-1.75 1.75-1.75h11.5a1.75 1.75 0 1 1 0 3.5h-11.5A1.75 1.75 0 0 1 41 26.25Z" clipRule="evenodd" />
        <path fill="#508FD6" fillRule="evenodd" d="M41 34.25c0-.967.783-1.75 1.75-1.75h11.5a1.75 1.75 0 1 1 0 3.5h-11.5A1.75 1.75 0 0 1 41 34.25Z" clipRule="evenodd" />
        <path fill="#3770C3" fillRule="evenodd" d="M41 42.25c0-.967.783-1.75 1.75-1.75h11.5a1.75 1.75 0 1 1 0 3.5h-11.5A1.75 1.75 0 0 1 41 42.25Z" clipRule="evenodd" />
        <path fill="#3770C3" d="M0 18.5a3 3 0 0 1 3-3h30a3 3 0 0 1 3 3v30a3 3 0 0 1-3 3H3a3 3 0 0 1-3-3v-30Z" />
        <path fill="#fff" d="M10.924 43.043 7 23h3.404l2.461 13.768L15.873 23h3.951l2.885 14 2.53-14h3.335l-3.992 20.043h-3.527l-3.282-14.984-3.253 14.984h-3.596Z" />
      </g>
      <defs>
        <clipPath id="a">
          <path fill="#fff" d="M0 0h64v64H0z" />
        </clipPath>
      </defs>
    </svg>
  ),
  sunni: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width={404} height={328} fill="none">
      <g filter="url(#a)">
        <circle cx={244} cy={105} r={61} fill="#FFC701" fillOpacity={0.5} />
      </g>
      <circle cx={244} cy={107} r={58} fill="url(#b)" stroke="url(#c)" strokeWidth={2} />
      <mask
        id="e"
        width={118}
        height={85}
        x={185}
        y={81}
        maskUnits="userSpaceOnUse"
        style={{
          maskType: "alpha",
        }}
      >
        <path fill="url(#d)" d="M303 107c0 32.585-26.415 59-59 59s-59-26.415-59-59 21.915-25 54.5-25 63.5-7.585 63.5 25Z" />
      </mask>
      <g filter="url(#f)" mask="url(#e)">
        <path
          fill="#E18700"
          d="M198 95c36.451 0 66 29.549 66 66 0 3.748-.313 7.423-.913 11H290.5v.006c.166-.002.333-.006.5-.006 23.748 0 43 19.252 43 43s-19.252 43-43 43c-.167 0-.334-.005-.5-.007V258h-151l.005-.007c-.168.002-.336.007-.505.007-23.748 0-43-19.252-43-43 0-21.707 16.085-39.655 36.985-42.58A66.425 66.425 0 0 1 132 161c0-36.451 29.549-66 66-66Z"
        />
      </g>
      <foreignObject width={338} height={261} x={0} y={67}>
        <div
          xmlns="http://www.w3.org/1999/xhtml"
          style={{
            backdropFilter: "blur(7px)",
            clipPath: "url(#g)",
            height: "100%",
            width: "100%",
          }}
        />
      </foreignObject>
      <g data-figma-bg-blur-radius={14} filter="url(#h)">
        <mask id="k" fill="#fff">
          <path d="M189 102c36.451 0 66 29.549 66 66 0 3.053-.208 6.057-.609 9H265.5v.006c.166-.002.333-.006.5-.006 23.748 0 43 19.252 43 43s-19.252 43-43 43c-.167 0-.334-.005-.5-.007V263h-151l.005-.007c-.168.002-.336.007-.505.007-23.748 0-43-19.252-43-43s19.252-43 43-43c3.361 0 6.632.387 9.771 1.116A66.414 66.414 0 0 1 123 168c0-36.451 29.549-66 66-66Z" />
        </mask>
        <path
          fill="url(#i)"
          d="M189 102c36.451 0 66 29.549 66 66 0 3.053-.208 6.057-.609 9H265.5v.006c.166-.002.333-.006.5-.006 23.748 0 43 19.252 43 43s-19.252 43-43 43c-.167 0-.334-.005-.5-.007V263h-151l.005-.007c-.168.002-.336.007-.505.007-23.748 0-43-19.252-43-43s19.252-43 43-43c3.361 0 6.632.387 9.771 1.116A66.414 66.414 0 0 1 123 168c0-36.451 29.549-66 66-66Z"
          shapeRendering="crispEdges"
        />
        <path
          fill="url(#j)"
          d="m254.391 177-1.982-.27a1.999 1.999 0 0 0 1.982 2.27v-2Zm11.109 0h2a2 2 0 0 0-2-2v2Zm0 .006h-2a2 2 0 0 0 2.023 2l-.023-2Zm0 85.987.023-2a1.999 1.999 0 0 0-2.023 2h2Zm0 .007v2a2 2 0 0 0 2-2h-2Zm-151 0-1.627-1.162A2 2 0 0 0 114.5 265v-2Zm.005-.007 1.627 1.163a2.001 2.001 0 0 0-1.65-3.163l.023 2Zm9.266-84.877-.453 1.948a1.998 1.998 0 0 0 2.429-2.252l-1.976.304ZM189 102v2c35.346 0 64 28.654 64 64h4c0-37.555-30.445-68-68-68v2Zm66 66h-2c0 2.962-.202 5.876-.591 8.73l1.982.27 1.981.27c.414-3.031.628-6.126.628-9.27h-2Zm-.609 9v2H265.5v-4h-11.109v2Zm11.109 0h-2v.006h4V177h-2Zm0 .006.023 2c.202-.003.323-.006.477-.006v-4c-.179 0-.392.004-.523.006l.023 2Zm.5-.006v2c22.644 0 41 18.356 41 41h4c0-24.853-20.147-45-45-45v2Zm43 43h-2c0 22.644-18.356 41-41 41v4c24.853 0 45-20.147 45-45h-2Zm-43 43v-2c-.161 0-.253-.004-.477-.007l-.023 2-.023 2c.109.001.35.007.523.007v-2Zm-.5-.007h-2V263h4v-.007h-2Zm0 .007v-2h-151v4h151v-2Zm-151 0 1.627 1.162.005-.006-1.627-1.163-1.628-1.162-.004.007L114.5 263Zm.005-.007-.023-2c-.223.003-.321.007-.482.007v4c.176 0 .415-.006.528-.007l-.023-2ZM114 263v-2c-22.644 0-41-18.356-41-41h-4c0 24.853 20.147 45 45 45v-2Zm-43-43h2c0-22.644 18.356-41 41-41v-4c-24.853 0-45 20.147-45 45h2Zm43-43v2c3.207 0 6.326.369 9.318 1.064l.453-1.948.452-1.948A45.153 45.153 0 0 0 114 175v2Zm9.771 1.116 1.976-.304A64.49 64.49 0 0 1 125 168h-4c0 3.542.271 7.022.794 10.42l1.977-.304ZM123 168h2c0-35.346 28.654-64 64-64v-4c-37.555 0-68 30.445-68 68h2Z"
          mask="url(#k)"
        />
      </g>
      <foreignObject width={272} height={216.353} x={132} y={67}>
        <div
          xmlns="http://www.w3.org/1999/xhtml"
          style={{
            backdropFilter: "blur(7px)",
            clipPath: "url(#l)",
            height: "100%",
            width: "100%",
          }}
        />
      </foreignObject>
      <g data-figma-bg-blur-radius={14} filter="url(#m)">
        <mask id="p" fill="#fff">
          <path d="M288.277 102c26.343 0 47.697 21.355 47.698 47.697 0 2.207-.151 4.378-.441 6.505h8.029v.004c.121-.001.241-.004.362-.004 17.162 0 31.075 13.913 31.075 31.075 0 17.163-13.913 31.075-31.075 31.076-.121 0-.241-.004-.362-.005v.005H234.437l.002-.005c-.121.001-.242.005-.364.005C216.913 218.352 203 204.44 203 187.277c0-17.162 13.913-31.075 31.075-31.075 2.429 0 4.793.278 7.062.805a48 48 0 0 1-.557-7.31c0-26.342 21.355-47.697 47.697-47.697Z" />
        </mask>
        <path
          fill="url(#n)"
          d="M288.277 102c26.343 0 47.697 21.355 47.698 47.697 0 2.207-.151 4.378-.441 6.505h8.029v.004c.121-.001.241-.004.362-.004 17.162 0 31.075 13.913 31.075 31.075 0 17.163-13.913 31.075-31.075 31.076-.121 0-.241-.004-.362-.005v.005H234.437l.002-.005c-.121.001-.242.005-.364.005C216.913 218.352 203 204.44 203 187.277c0-17.162 13.913-31.075 31.075-31.075 2.429 0 4.793.278 7.062.805a48 48 0 0 1-.557-7.31c0-26.342 21.355-47.697 47.697-47.697Z"
          shapeRendering="crispEdges"
        />
        <path
          fill="url(#o)"
          d="M288.277 102v-2 2Zm47.698 47.697h2-2Zm-.441 6.505-1.981-.27a1.998 1.998 0 0 0 1.981 2.27v-2Zm8.029 0h2a2 2 0 0 0-2-2v2Zm0 .004h-2a1.999 1.999 0 0 0 2.023 2l-.023-2Zm.362-.004v-2 2ZM375 187.277h2-2Zm-31.075 31.076v2-2Zm-.362-.005.023-2a2.001 2.001 0 0 0-2.023 2h2Zm0 .005v2a2 2 0 0 0 2-2h-2Zm-109.126 0-1.715-1.029a1.999 1.999 0 0 0 1.715 3.029v-2Zm.002-.005 1.715 1.029a2 2 0 0 0-1.737-3.029l.022 2Zm-.364.005v2-2ZM203 187.277h-2 2Zm31.075-31.075v-2 2Zm7.062.805-.453 1.948a2 2 0 0 0 2.429-2.252l-1.976.304Zm-.557-7.31h-2 2ZM288.277 102v2c25.238 0 45.697 20.459 45.698 45.697h4C337.974 122.25 315.724 100 288.277 100v2Zm47.698 47.697h-2c0 2.116-.145 4.197-.422 6.235l1.981.27 1.982.27c.302-2.215.459-4.477.459-6.775h-2Zm-.441 6.505v2h8.029v-4h-8.029v2Zm8.029 0h-2v.004h4v-.004h-2Zm0 .004.023 2c.147-.002.233-.004.339-.004v-4c-.135 0-.291.003-.384.004l.022 2Zm.362-.004v2c16.058 0 29.075 13.017 29.075 29.075h4c0-18.267-14.808-33.075-33.075-33.075v2ZM375 187.277h-2c0 16.058-13.017 29.075-29.075 29.076v4c18.267-.001 33.075-14.809 33.075-33.076h-2Zm-31.075 31.076v-2c-.114 0-.163-.003-.339-.005l-.023 2-.022 2c.064 0 .257.005.384.005v-2Zm-.362-.005h-2v.005h4v-.005h-2Zm0 .005v-2H234.437v4h109.126v-2Zm-109.126 0 1.715 1.029.002-.005-1.715-1.029-1.715-1.029-.002.005 1.715 1.029Zm.002-.005-.022-2c-.175.002-.228.005-.342.005v4c.129 0 .32-.005.387-.005l-.023-2Zm-.364.005v-2C218.017 216.352 205 203.335 205 187.277h-4c0 18.267 14.808 33.075 33.075 33.076v-2ZM203 187.277h2c0-16.058 13.017-29.075 29.075-29.075v-4c-18.267 0-33.075 14.808-33.075 33.075h2Zm31.075-31.075v2c2.276 0 4.489.26 6.609.753l.453-1.948.452-1.948a33.135 33.135 0 0 0-7.514-.857v2Zm7.062.805 1.976-.304a46.076 46.076 0 0 1-.533-7.006h-4c0 2.588.198 5.131.58 7.614l1.977-.304Zm-.557-7.31h2c0-25.238 20.459-45.697 45.697-45.697v-4c-27.447 0-49.697 22.25-49.697 49.697h2Z"
          mask="url(#p)"
        />
      </g>
      <defs>
        <linearGradient id="b" x1={244} x2={244} y1={48} y2={166} gradientUnits="userSpaceOnUse">
          <stop stopColor="#FFE600" />
          <stop offset={1} stopColor="#FF7A00" />
        </linearGradient>
        <linearGradient id="c" x1={244} x2={244} y1={48} y2={166} gradientUnits="userSpaceOnUse">
          <stop stopColor="#fff" />
          <stop offset={1} stopColor="#fff" stopOpacity={0} />
        </linearGradient>
        <linearGradient id="d" x1={244} x2={244} y1={48} y2={166} gradientUnits="userSpaceOnUse">
          <stop stopColor="#FFD600" />
          <stop offset={1} stopColor="#FF7A00" />
        </linearGradient>
        <linearGradient id="i" x1={190} x2={190} y1={50.5} y2={285.5} gradientUnits="userSpaceOnUse">
          <stop stopColor="#fff" />
          <stop offset={1} stopColor="#fff" stopOpacity={0} />
        </linearGradient>
        <linearGradient id="j" x1={190} x2={190} y1={102} y2={263} gradientUnits="userSpaceOnUse">
          <stop stopColor="#fff" />
          <stop offset={1} stopColor="#fff" stopOpacity={0} />
        </linearGradient>
        <linearGradient id="n" x1={297} x2={297} y1={105.5} y2={231.5} gradientUnits="userSpaceOnUse">
          <stop stopColor="#fff" />
          <stop offset={1} stopColor="#fff" stopOpacity={0} />
        </linearGradient>
        <linearGradient id="o" x1={289} x2={289} y1={102} y2={218.353} gradientUnits="userSpaceOnUse">
          <stop stopColor="#fff" />
          <stop offset={1} stopColor="#fff" stopOpacity={0} />
        </linearGradient>
        <filter id="a" width={210} height={210} x={139} y={0} colorInterpolationFilters="sRGB" filterUnits="userSpaceOnUse">
          <feFlood floodOpacity={0} result="BackgroundImageFix" />
          <feBlend in="SourceGraphic" in2="BackgroundImageFix" result="shape" />
          <feGaussianBlur result="effect1_foregroundBlur_102_2" stdDeviation={22} />
        </filter>
        <filter id="f" width={266} height={191} x={82} y={81} colorInterpolationFilters="sRGB" filterUnits="userSpaceOnUse">
          <feFlood floodOpacity={0} result="BackgroundImageFix" />
          <feBlend in="SourceGraphic" in2="BackgroundImageFix" result="shape" />
          <feGaussianBlur result="effect1_foregroundBlur_102_2" stdDeviation={7} />
        </filter>
        <filter id="h" width={338} height={261} x={0} y={67} colorInterpolationFilters="sRGB" filterUnits="userSpaceOnUse">
          <feFlood floodOpacity={0} result="BackgroundImageFix" />
          <feColorMatrix in="SourceAlpha" result="hardAlpha" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 127 0" />
          <feOffset dx={-21} dy={15} />
          <feGaussianBlur stdDeviation={25} />
          <feComposite in2="hardAlpha" operator="out" />
          <feColorMatrix values="0 0 0 0 0.113725 0 0 0 0 0.14902 0 0 0 0 0.27451 0 0 0 0.25 0" />
          <feBlend in2="BackgroundImageFix" result="effect1_dropShadow_102_2" />
          <feBlend in="SourceGraphic" in2="effect1_dropShadow_102_2" result="shape" />
        </filter>
        <filter id="m" width={272} height={216.353} x={132} y={67} colorInterpolationFilters="sRGB" filterUnits="userSpaceOnUse">
          <feFlood floodOpacity={0} result="BackgroundImageFix" />
          <feColorMatrix in="SourceAlpha" result="hardAlpha" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 127 0" />
          <feOffset dx={-21} dy={15} />
          <feGaussianBlur stdDeviation={25} />
          <feComposite in2="hardAlpha" operator="out" />
          <feColorMatrix values="0 0 0 0 0.113438 0 0 0 0 0.148981 0 0 0 0 0.275 0 0 0 0.25 0" />
          <feBlend in2="BackgroundImageFix" result="effect1_dropShadow_102_2" />
          <feBlend in="SourceGraphic" in2="effect1_dropShadow_102_2" result="shape" />
        </filter>
        <clipPath id="g" transform="translate(0 -67)">
          <path d="M189 102c36.451 0 66 29.549 66 66 0 3.053-.208 6.057-.609 9H265.5v.006c.166-.002.333-.006.5-.006 23.748 0 43 19.252 43 43s-19.252 43-43 43c-.167 0-.334-.005-.5-.007V263h-151l.005-.007c-.168.002-.336.007-.505.007-23.748 0-43-19.252-43-43s19.252-43 43-43c3.361 0 6.632.387 9.771 1.116A66.414 66.414 0 0 1 123 168c0-36.451 29.549-66 66-66Z" />
        </clipPath>
        <clipPath id="l" transform="translate(-132 -67)">
          <path d="M288.277 102c26.343 0 47.697 21.355 47.698 47.697 0 2.207-.151 4.378-.441 6.505h8.029v.004c.121-.001.241-.004.362-.004 17.162 0 31.075 13.913 31.075 31.075 0 17.163-13.913 31.075-31.075 31.076-.121 0-.241-.004-.362-.005v.005H234.437l.002-.005c-.121.001-.242.005-.364.005C216.913 218.352 203 204.44 203 187.277c0-17.162 13.913-31.075 31.075-31.075 2.429 0 4.793.278 7.062.805a48 48 0 0 1-.557-7.31c0-26.342 21.355-47.697 47.697-47.697Z" />
        </clipPath>
      </defs>
    </svg>
  ),
  cloudyy: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width={444} height={296} fill="none">
      <foreignObject width={272} height={216.353} x={0} y={0}>
        <div
          xmlns="http://www.w3.org/1999/xhtml"
          style={{
            backdropFilter: "blur(7px)",
            clipPath: "url(#a)",
            height: "100%",
            width: "100%",
          }}
        />
      </foreignObject>
      <g data-figma-bg-blur-radius={14} filter="url(#b)">
        <mask id="e" fill="#fff">
          <path
            fillRule="evenodd"
            d="m211.563 151.351.361.002c17.163 0 31.076-13.913 31.076-31.076 0-17.162-13.913-31.075-31.076-31.075-.12 0-.241 0-.361.002v-.002h-8.028c.29-2.127.44-4.298.44-6.504C203.975 56.355 182.62 35 156.277 35c-26.342 0-47.697 21.355-47.697 47.698 0 2.485.19 4.926.556 7.31a31.16 31.16 0 0 0-7.06-.806C84.913 89.202 71 103.115 71 120.277c0 17.163 13.913 31.076 31.076 31.076l.363-.002-.002.002h109.126v-.002Z"
            clipRule="evenodd"
          />
        </mask>
        <path
          fill="url(#c)"
          fillRule="evenodd"
          d="m211.563 151.351.361.002c17.163 0 31.076-13.913 31.076-31.076 0-17.162-13.913-31.075-31.076-31.075-.12 0-.241 0-.361.002v-.002h-8.028c.29-2.127.44-4.298.44-6.504C203.975 56.355 182.62 35 156.277 35c-26.342 0-47.697 21.355-47.697 47.698 0 2.485.19 4.926.556 7.31a31.16 31.16 0 0 0-7.06-.806C84.913 89.202 71 103.115 71 120.277c0 17.163 13.913 31.076 31.076 31.076l.363-.002-.002.002h109.126v-.002Z"
          clipRule="evenodd"
          shapeRendering="crispEdges"
        />
        <path
          fill="url(#d)"
          d="m211.563 151.351.023-2a2 2 0 0 0-2.023 2h2Zm0-62.147h-2a2 2 0 0 0 2.023 2l-.023-2Zm0-.002h2a2 2 0 0 0-2-2v2Zm-8.028 0-1.982-.27a2.003 2.003 0 0 0 1.982 2.27v-2Zm-94.399.805-.452 1.948a2 2 0 0 0 2.429-2.252l-1.977.304Zm-6.697 61.344 1.569 1.239a2 2 0 0 0-1.592-3.239l.023 2Zm-.002.002-1.57-1.239a2 2 0 0 0 1.57 3.239v-2Zm109.126 0v2a2 2 0 0 0 2-2h-2Zm-.023 1.998.384.002v-4l-.338-.002-.046 4Zm.384.002c18.268 0 33.076-14.809 33.076-33.076h-4c0 16.058-13.018 29.076-29.076 29.076v4ZM245 120.277c0-18.267-14.808-33.075-33.076-33.075v4c16.058 0 29.076 13.017 29.076 29.075h4Zm-33.076-33.075c-.128 0-.256 0-.384.002l.046 4c.113-.002.225-.002.338-.002v-4Zm-2.361 2v.002h4v-.002h-4Zm-6.028 2h8.028v-4h-8.028v4Zm1.982-1.73a50.15 50.15 0 0 0 .458-6.774h-4c0 2.115-.144 4.196-.422 6.234l3.964.54Zm.458-6.774C205.975 55.25 183.724 33 156.277 33v4c25.238 0 45.698 20.46 45.698 45.698h4ZM156.277 33c-27.447 0-49.697 22.25-49.697 49.698h4C110.58 57.46 131.039 37 156.277 37v-4ZM106.58 82.698c0 2.587.198 5.13.58 7.613l3.953-.608a46.076 46.076 0 0 1-.533-7.005h-4Zm3.009 5.361a33.172 33.172 0 0 0-7.513-.857v4c2.275 0 4.487.26 6.608.754l.905-3.897Zm-7.513-.857C83.809 87.202 69 102.01 69 120.277h4c0-16.058 13.018-29.075 29.076-29.075v-4ZM69 120.277c0 18.267 14.808 33.076 33.076 33.076v-4C86.018 149.353 73 136.335 73 120.277h-4Zm33.076 33.076.385-.002-.045-4-.34.002v4Zm1.931-.761.001-.002-3.139-2.478-.002.002 3.14 2.478Zm107.556-3.239H102.437v4h109.126v-4Zm-2 1.998v.002h4v-.002h-4Z"
          mask="url(#e)"
        />
      </g>
      <foreignObject width={338} height={261} x={40} y={35}>
        <div
          xmlns="http://www.w3.org/1999/xhtml"
          style={{
            backdropFilter: "blur(7px)",
            clipPath: "url(#f)",
            height: "100%",
            width: "100%",
          }}
        />
      </foreignObject>
      <g data-figma-bg-blur-radius={14} filter="url(#g)">
        <mask id="j" fill="#fff">
          <path
            fillRule="evenodd"
            d="M305.5 230.997c.166.002.333.003.5.003 23.748 0 43-19.252 43-43s-19.252-43-43-43c-.167 0-.334.001-.5.003V145h-11.109a66.47 66.47 0 0 0 .609-9c0-36.45-29.549-66-66-66s-66 29.55-66 66c0 3.439.263 6.817.77 10.115A43.142 43.142 0 0 0 154 145c-23.748 0-43 19.252-43 43s19.252 43 43 43c.168 0 .335-.001.502-.003l-.002.003h151v-.003Z"
            clipRule="evenodd"
          />
        </mask>
        <path
          fill="url(#h)"
          fillRule="evenodd"
          d="M305.5 230.997c.166.002.333.003.5.003 23.748 0 43-19.252 43-43s-19.252-43-43-43c-.167 0-.334.001-.5.003V145h-11.109a66.47 66.47 0 0 0 .609-9c0-36.45-29.549-66-66-66s-66 29.55-66 66c0 3.439.263 6.817.77 10.115A43.142 43.142 0 0 0 154 145c-23.748 0-43 19.252-43 43s19.252 43 43 43c.168 0 .335-.001.502-.003l-.002.003h151v-.003Z"
          clipRule="evenodd"
          shapeRendering="crispEdges"
        />
        <path
          fill="url(#i)"
          d="m305.5 230.997.023-2a1.999 1.999 0 0 0-2.023 2h2Zm0-85.994h-2a2 2 0 0 0 2.023 2l-.023-2Zm0-.003h2a2 2 0 0 0-2-2v2Zm-11.109 0-1.981-.27a1.999 1.999 0 0 0 1.981 2.27v-2Zm-130.621 1.115-.452 1.948a2.001 2.001 0 0 0 2.429-2.252l-1.977.304Zm-9.268 84.882 1.57 1.24a2.001 2.001 0 0 0-1.593-3.24l.023 2Zm-.002.003-1.569-1.24A2 2 0 0 0 154.5 233v-2Zm151 0v2a2 2 0 0 0 2-2h-2Zm-.023 1.997c.174.002.348.003.523.003v-4l-.477-.003-.046 4ZM306 233c24.853 0 45-20.147 45-45h-4c0 22.644-18.356 41-41 41v4Zm45-45c0-24.853-20.147-45-45-45v4c22.644 0 41 18.356 41 41h4Zm-45-45c-.174 0-.349.001-.523.003l.046 4L306 147v-4Zm-2.5 2v.003h4V145h-4Zm-9.109 2H305.5v-4h-11.109v4Zm1.982-1.73A68.5 68.5 0 0 0 297 136h-4c0 2.962-.201 5.876-.59 8.73l3.963.54ZM297 136c0-37.555-30.445-68-68-68v4c35.346 0 64 28.654 64 64h4Zm-68-68c-37.555 0-68 30.445-68 68h4c0-35.346 28.654-64 64-64v-4Zm-68 68c0 3.542.271 7.021.793 10.419l3.954-.608A64.477 64.477 0 0 1 165 136h-4Zm3.223 8.167A45.103 45.103 0 0 0 154 143v4c3.207 0 6.326.368 9.318 1.063l.905-3.896ZM154 143c-24.853 0-45 20.147-45 45h4c0-22.644 18.356-41 41-41v-4Zm-45 45c0 24.853 20.147 45 45 45v-4c-22.644 0-41-18.356-41-41h-4Zm45 45 .525-.003-.046-4c-.159.002-.319.003-.479.003v4Zm2.069-.76.003-.003-3.139-2.48-.002.003 3.138 2.48ZM305.5 229h-151v4h151v-4Zm-2 1.997V231h4v-.003h-4Z"
          mask="url(#j)"
        />
      </g>
      <foreignObject width={272} height={216.353} x={172} y={35}>
        <div
          xmlns="http://www.w3.org/1999/xhtml"
          style={{
            backdropFilter: "blur(7px)",
            clipPath: "url(#k)",
            height: "100%",
            width: "100%",
          }}
        />
      </foreignObject>
      <g data-figma-bg-blur-radius={14} filter="url(#l)">
        <mask id="o" fill="#fff">
          <path
            fillRule="evenodd"
            d="m383.563 186.351.361.002c17.163 0 31.076-13.913 31.076-31.076 0-17.162-13.913-31.075-31.076-31.075-.12 0-.241 0-.361.002v-.002h-8.028c.29-2.127.44-4.298.44-6.505C375.975 91.355 354.62 70 328.277 70c-26.342 0-47.697 21.355-47.697 47.697 0 2.486.19 4.927.556 7.31a31.166 31.166 0 0 0-7.06-.805c-17.163 0-31.076 13.913-31.076 31.075 0 17.163 13.913 31.076 31.076 31.076l.363-.002-.002.002h109.126v-.002Z"
            clipRule="evenodd"
          />
        </mask>
        <path
          fill="url(#m)"
          fillRule="evenodd"
          d="m383.563 186.351.361.002c17.163 0 31.076-13.913 31.076-31.076 0-17.162-13.913-31.075-31.076-31.075-.12 0-.241 0-.361.002v-.002h-8.028c.29-2.127.44-4.298.44-6.505C375.975 91.355 354.62 70 328.277 70c-26.342 0-47.697 21.355-47.697 47.697 0 2.486.19 4.927.556 7.31a31.166 31.166 0 0 0-7.06-.805c-17.163 0-31.076 13.913-31.076 31.075 0 17.163 13.913 31.076 31.076 31.076l.363-.002-.002.002h109.126v-.002Z"
          clipRule="evenodd"
          shapeRendering="crispEdges"
        />
        <path
          fill="url(#n)"
          d="m383.563 186.351.023-2a2 2 0 0 0-2.023 2h2Zm0-62.147h-2a2 2 0 0 0 2.023 2l-.023-2Zm0-.002h2a2 2 0 0 0-2-2v2Zm-8.028 0-1.982-.271a2.003 2.003 0 0 0 1.982 2.271v-2Zm-94.399.805-.452 1.948a1.998 1.998 0 0 0 2.429-2.252l-1.977.304Zm-6.697 61.344 1.569 1.239a2 2 0 0 0-1.592-3.239l.023 2Zm-.002.002-1.57-1.239a2 2 0 0 0 1.57 3.239v-2Zm109.126 0v2a2 2 0 0 0 2-2h-2Zm-.023 1.998.384.002v-4l-.338-.002-.046 4Zm.384.002c18.268 0 33.076-14.809 33.076-33.076h-4c0 16.058-13.018 29.076-29.076 29.076v4ZM417 155.277c0-18.267-14.808-33.075-33.076-33.075v4c16.058 0 29.076 13.017 29.076 29.075h4Zm-33.076-33.075c-.128 0-.256 0-.384.002l.046 4c.113-.002.225-.002.338-.002v-4Zm-2.361 2v.002h4v-.002h-4Zm-6.028 2h8.028v-4h-8.028v4Zm1.982-1.73c.302-2.216.458-4.478.458-6.775h-4c0 2.116-.144 4.197-.422 6.234l3.964.541Zm.458-6.775C377.975 90.25 355.724 68 328.277 68v4c25.238 0 45.698 20.46 45.698 45.697h4ZM328.277 68c-27.447 0-49.697 22.25-49.697 49.697h4C282.58 92.46 303.039 72 328.277 72v-4Zm-49.697 49.697c0 2.588.198 5.131.58 7.614l3.953-.608a46.066 46.066 0 0 1-.533-7.006h-4Zm3.009 5.362a33.183 33.183 0 0 0-7.513-.857v4c2.275 0 4.487.261 6.608.753l.905-3.896Zm-7.513-.857c-18.268 0-33.076 14.808-33.076 33.075h4c0-16.058 13.018-29.075 29.076-29.075v-4ZM241 155.277c0 18.267 14.808 33.076 33.076 33.076v-4c-16.058 0-29.076-13.018-29.076-29.076h-4Zm33.076 33.076.385-.002-.045-4-.34.002v4Zm1.931-.761.001-.002-3.139-2.478-.002.002 3.14 2.478Zm107.556-3.239H274.437v4h109.126v-4Zm-2 1.998v.002h4v-.002h-4Z"
          mask="url(#o)"
        />
      </g>
      <defs>
        <linearGradient id="c" x1={165} x2={165} y1={38.5} y2={164.5} gradientUnits="userSpaceOnUse">
          <stop stopColor="#fff" />
          <stop offset={1} stopColor="#fff" stopOpacity={0} />
        </linearGradient>
        <linearGradient id="d" x1={157} x2={157} y1={35} y2={151.353} gradientUnits="userSpaceOnUse">
          <stop stopColor="#fff" />
          <stop offset={1} stopColor="#fff" stopOpacity={0} />
        </linearGradient>
        <linearGradient id="h" x1={230} x2={230} y1={18.5} y2={253.5} gradientUnits="userSpaceOnUse">
          <stop stopColor="#fff" />
          <stop offset={1} stopColor="#fff" stopOpacity={0} />
        </linearGradient>
        <linearGradient id="i" x1={230} x2={230} y1={70} y2={231} gradientUnits="userSpaceOnUse">
          <stop stopColor="#fff" />
          <stop offset={1} stopColor="#fff" stopOpacity={0} />
        </linearGradient>
        <linearGradient id="m" x1={337} x2={337} y1={73.5} y2={199.5} gradientUnits="userSpaceOnUse">
          <stop stopColor="#fff" />
          <stop offset={1} stopColor="#fff" stopOpacity={0} />
        </linearGradient>
        <linearGradient id="n" x1={329} x2={329} y1={70} y2={186.353} gradientUnits="userSpaceOnUse">
          <stop stopColor="#fff" />
          <stop offset={1} stopColor="#fff" stopOpacity={0} />
        </linearGradient>
        <clipPath id="a">
          <path
            fillRule="evenodd"
            d="m211.563 151.351.361.002c17.163 0 31.076-13.913 31.076-31.076 0-17.162-13.913-31.075-31.076-31.075-.12 0-.241 0-.361.002v-.002h-8.028c.29-2.127.44-4.298.44-6.504C203.975 56.355 182.62 35 156.277 35c-26.342 0-47.697 21.355-47.697 47.698 0 2.485.19 4.926.556 7.31a31.16 31.16 0 0 0-7.06-.806C84.913 89.202 71 103.115 71 120.277c0 17.163 13.913 31.076 31.076 31.076l.363-.002-.002.002h109.126v-.002Z"
            clipRule="evenodd"
          />
        </clipPath>
        <clipPath id="f" transform="translate(-40 -35)">
          <path
            fillRule="evenodd"
            d="M305.5 230.997c.166.002.333.003.5.003 23.748 0 43-19.252 43-43s-19.252-43-43-43c-.167 0-.334.001-.5.003V145h-11.109a66.47 66.47 0 0 0 .609-9c0-36.45-29.549-66-66-66s-66 29.55-66 66c0 3.439.263 6.817.77 10.115A43.142 43.142 0 0 0 154 145c-23.748 0-43 19.252-43 43s19.252 43 43 43c.168 0 .335-.001.502-.003l-.002.003h151v-.003Z"
            clipRule="evenodd"
          />
        </clipPath>
        <clipPath id="k" transform="translate(-172 -35)">
          <path
            fillRule="evenodd"
            d="m383.563 186.351.361.002c17.163 0 31.076-13.913 31.076-31.076 0-17.162-13.913-31.075-31.076-31.075-.12 0-.241 0-.361.002v-.002h-8.028c.29-2.127.44-4.298.44-6.505C375.975 91.355 354.62 70 328.277 70c-26.342 0-47.697 21.355-47.697 47.697 0 2.486.19 4.927.556 7.31a31.166 31.166 0 0 0-7.06-.805c-17.163 0-31.076 13.913-31.076 31.075 0 17.163 13.913 31.076 31.076 31.076l.363-.002-.002.002h109.126v-.002Z"
            clipRule="evenodd"
          />
        </clipPath>
        <filter id="b" width={272} height={216.353} x={0} y={0} colorInterpolationFilters="sRGB" filterUnits="userSpaceOnUse">
          <feFlood floodOpacity={0} result="BackgroundImageFix" />
          <feColorMatrix in="SourceAlpha" result="hardAlpha" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 127 0" />
          <feOffset dx={-21} dy={15} />
          <feGaussianBlur stdDeviation={25} />
          <feComposite in2="hardAlpha" operator="out" />
          <feColorMatrix values="0 0 0 0 0.113438 0 0 0 0 0.148981 0 0 0 0 0.275 0 0 0 0.25 0" />
          <feBlend in2="BackgroundImageFix" result="effect1_dropShadow_2_126" />
          <feBlend in="SourceGraphic" in2="effect1_dropShadow_2_126" result="shape" />
        </filter>
        <filter id="g" width={338} height={261} x={40} y={35} colorInterpolationFilters="sRGB" filterUnits="userSpaceOnUse">
          <feFlood floodOpacity={0} result="BackgroundImageFix" />
          <feColorMatrix in="SourceAlpha" result="hardAlpha" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 127 0" />
          <feOffset dx={-21} dy={15} />
          <feGaussianBlur stdDeviation={25} />
          <feComposite in2="hardAlpha" operator="out" />
          <feColorMatrix values="0 0 0 0 0.113725 0 0 0 0 0.14902 0 0 0 0 0.27451 0 0 0 0.25 0" />
          <feBlend in2="BackgroundImageFix" result="effect1_dropShadow_2_126" />
          <feBlend in="SourceGraphic" in2="effect1_dropShadow_2_126" result="shape" />
        </filter>
        <filter id="l" width={272} height={216.353} x={172} y={35} colorInterpolationFilters="sRGB" filterUnits="userSpaceOnUse">
          <feFlood floodOpacity={0} result="BackgroundImageFix" />
          <feColorMatrix in="SourceAlpha" result="hardAlpha" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 127 0" />
          <feOffset dx={-21} dy={15} />
          <feGaussianBlur stdDeviation={25} />
          <feComposite in2="hardAlpha" operator="out" />
          <feColorMatrix values="0 0 0 0 0.113438 0 0 0 0 0.148981 0 0 0 0 0.275 0 0 0 0.25 0" />
          <feBlend in2="BackgroundImageFix" result="effect1_dropShadow_2_126" />
          <feBlend in="SourceGraphic" in2="effect1_dropShadow_2_126" result="shape" />
        </filter>
      </defs>
    </svg>
  ),
  rainy: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width={404} height={314} fill="none">
      <foreignObject width={338} height={261} x={0} y={0}>
        <div
          xmlns="http://www.w3.org/1999/xhtml"
          style={{
            backdropFilter: "blur(7px)",
            clipPath: "url(#a)",
            height: "100%",
            width: "100%",
          }}
        />
      </foreignObject>
      <g data-figma-bg-blur-radius={14} filter="url(#b)">
        <mask id="e" fill="#fff">
          <path
            fillRule="evenodd"
            d="M265.5 195.997c.166.002.333.003.5.003 23.748 0 43-19.252 43-43s-19.252-43-43-43c-.167 0-.334.001-.5.003V110h-11.109a66.47 66.47 0 0 0 .609-9c0-36.45-29.549-66-66-66s-66 29.55-66 66c0 3.439.263 6.817.77 10.115A43.142 43.142 0 0 0 114 110c-23.748 0-43 19.252-43 43s19.252 43 43 43c.168 0 .335-.001.502-.003l-.002.003h151v-.003Z"
            clipRule="evenodd"
          />
        </mask>
        <path
          fill="url(#c)"
          fillRule="evenodd"
          d="M265.5 195.997c.166.002.333.003.5.003 23.748 0 43-19.252 43-43s-19.252-43-43-43c-.167 0-.334.001-.5.003V110h-11.109a66.47 66.47 0 0 0 .609-9c0-36.45-29.549-66-66-66s-66 29.55-66 66c0 3.439.263 6.817.77 10.115A43.142 43.142 0 0 0 114 110c-23.748 0-43 19.252-43 43s19.252 43 43 43c.168 0 .335-.001.502-.003l-.002.003h151v-.003Z"
          clipRule="evenodd"
          shapeRendering="crispEdges"
        />
        <path
          fill="url(#d)"
          d="m265.5 195.997.023-2a1.999 1.999 0 0 0-2.023 2h2Zm0-85.994h-2a2 2 0 0 0 2.023 2l-.023-2Zm0-.003h2a2 2 0 0 0-2-2v2Zm-11.109 0-1.981-.27a1.999 1.999 0 0 0 1.981 2.27v-2Zm-130.621 1.115-.452 1.948a2.001 2.001 0 0 0 2.429-2.252l-1.977.304Zm-9.268 84.882 1.57 1.24a2.001 2.001 0 0 0-1.593-3.24l.023 2Zm-.002.003-1.569-1.24A2 2 0 0 0 114.5 198v-2Zm151 0v2a2 2 0 0 0 2-2h-2Zm-.023 1.997c.174.002.348.003.523.003v-4l-.477-.003-.046 4ZM266 198c24.853 0 45-20.147 45-45h-4c0 22.644-18.356 41-41 41v4Zm45-45c0-24.853-20.147-45-45-45v4c22.644 0 41 18.356 41 41h4Zm-45-45c-.174 0-.349.001-.523.003l.046 4L266 112v-4Zm-2.5 2v.003h4V110h-4Zm-9.109 2H265.5v-4h-11.109v4Zm1.982-1.73A68.5 68.5 0 0 0 257 101h-4c0 2.962-.201 5.876-.59 8.73l3.963.54ZM257 101c0-37.555-30.445-68-68-68v4c35.346 0 64 28.654 64 64h4Zm-68-68c-37.555 0-68 30.445-68 68h4c0-35.346 28.654-64 64-64v-4Zm-68 68c0 3.542.271 7.021.793 10.419l3.954-.608A64.477 64.477 0 0 1 125 101h-4Zm3.223 8.167A45.103 45.103 0 0 0 114 108v4c3.207 0 6.326.368 9.318 1.063l.905-3.896ZM114 108c-24.853 0-45 20.147-45 45h4c0-22.644 18.356-41 41-41v-4Zm-45 45c0 24.853 20.147 45 45 45v-4c-22.644 0-41-18.356-41-41h-4Zm45 45 .525-.003-.046-4c-.159.002-.319.003-.479.003v4Zm2.069-.76.003-.003-3.139-2.48-.002.003 3.138 2.48ZM265.5 194h-151v4h151v-4Zm-2 1.997V196h4v-.003h-4Z"
          mask="url(#e)"
        />
      </g>
      <foreignObject width={272} height={216.353} x={132} y={0}>
        <div
          xmlns="http://www.w3.org/1999/xhtml"
          style={{
            backdropFilter: "blur(7px)",
            clipPath: "url(#f)",
            height: "100%",
            width: "100%",
          }}
        />
      </foreignObject>
      <g data-figma-bg-blur-radius={14} filter="url(#g)">
        <mask id="j" fill="#fff">
          <path
            fillRule="evenodd"
            d="m343.563 151.351.361.002c17.163 0 31.076-13.913 31.076-31.076 0-17.162-13.913-31.075-31.076-31.075-.12 0-.241 0-.361.002v-.002h-8.028c.29-2.127.44-4.298.44-6.504C335.975 56.355 314.62 35 288.277 35c-26.342 0-47.697 21.355-47.697 47.698 0 2.485.19 4.926.556 7.31a31.16 31.16 0 0 0-7.06-.806c-17.163 0-31.076 13.913-31.076 31.075 0 17.163 13.913 31.076 31.076 31.076l.363-.002-.002.002h109.126v-.002Z"
            clipRule="evenodd"
          />
        </mask>
        <path
          fill="url(#h)"
          fillRule="evenodd"
          d="m343.563 151.351.361.002c17.163 0 31.076-13.913 31.076-31.076 0-17.162-13.913-31.075-31.076-31.075-.12 0-.241 0-.361.002v-.002h-8.028c.29-2.127.44-4.298.44-6.504C335.975 56.355 314.62 35 288.277 35c-26.342 0-47.697 21.355-47.697 47.698 0 2.485.19 4.926.556 7.31a31.16 31.16 0 0 0-7.06-.806c-17.163 0-31.076 13.913-31.076 31.075 0 17.163 13.913 31.076 31.076 31.076l.363-.002-.002.002h109.126v-.002Z"
          clipRule="evenodd"
          shapeRendering="crispEdges"
        />
        <path
          fill="url(#i)"
          d="m343.563 151.351.023-2a2 2 0 0 0-2.023 2h2Zm0-62.147h-2a2 2 0 0 0 2.023 2l-.023-2Zm0-.002h2a2 2 0 0 0-2-2v2Zm-8.028 0-1.982-.27a2.003 2.003 0 0 0 1.982 2.27v-2Zm-94.399.805-.452 1.948a2 2 0 0 0 2.429-2.252l-1.977.304Zm-6.697 61.344 1.569 1.239a2 2 0 0 0-1.592-3.239l.023 2Zm-.002.002-1.57-1.239a2 2 0 0 0 1.57 3.239v-2Zm109.126 0v2a2 2 0 0 0 2-2h-2Zm-.023 1.998.384.002v-4l-.338-.002-.046 4Zm.384.002c18.268 0 33.076-14.809 33.076-33.076h-4c0 16.058-13.018 29.076-29.076 29.076v4ZM377 120.277c0-18.267-14.808-33.075-33.076-33.075v4c16.058 0 29.076 13.017 29.076 29.075h4Zm-33.076-33.075c-.128 0-.256 0-.384.002l.046 4c.113-.002.225-.002.338-.002v-4Zm-2.361 2v.002h4v-.002h-4Zm-6.028 2h8.028v-4h-8.028v4Zm1.982-1.73a50.15 50.15 0 0 0 .458-6.774h-4c0 2.115-.144 4.196-.422 6.234l3.964.54Zm.458-6.774C337.975 55.25 315.724 33 288.277 33v4c25.238 0 45.698 20.46 45.698 45.698h4ZM288.277 33c-27.447 0-49.697 22.25-49.697 49.698h4C242.58 57.46 263.039 37 288.277 37v-4ZM238.58 82.698c0 2.587.198 5.13.58 7.613l3.953-.608a46.076 46.076 0 0 1-.533-7.005h-4Zm3.009 5.361a33.172 33.172 0 0 0-7.513-.857v4c2.275 0 4.487.26 6.608.754l.905-3.897Zm-7.513-.857c-18.268 0-33.076 14.808-33.076 33.075h4c0-16.058 13.018-29.075 29.076-29.075v-4ZM201 120.277c0 18.267 14.808 33.076 33.076 33.076v-4c-16.058 0-29.076-13.018-29.076-29.076h-4Zm33.076 33.076.385-.002-.045-4-.34.002v4Zm1.931-.761.001-.002-3.139-2.478-.002.002 3.14 2.478Zm107.556-3.239H234.437v4h109.126v-4Zm-2 1.998v.002h4v-.002h-4Z"
          mask="url(#j)"
        />
      </g>
      <g strokeWidth={2} filter="url(#k)">
        <path fill="url(#l)" stroke="url(#m)" d="M165.708 169.3c3.302-4.238 10.09-2.173 10.473 3.185l1.174 16.434c.608 8.529-7.525 15.017-15.705 12.529-8.18-2.489-11.322-12.407-6.067-19.151l10.125-12.997Z" />
        <path fill="url(#n)" stroke="url(#o)" d="M223.216 169.3c3.301-4.238 10.089-2.173 10.472 3.185l1.174 16.434c.609 8.529-7.525 15.017-15.705 12.529-8.18-2.489-11.322-12.407-6.067-19.151l10.126-12.997Z" />
        <path fill="url(#p)" stroke="url(#q)" d="M280.723 169.3c3.302-4.238 10.09-2.173 10.473 3.185l1.173 16.434c.609 8.529-7.524 15.017-15.705 12.529-8.179-2.489-11.321-12.407-6.067-19.151l10.126-12.997Z" />
      </g>
      <g strokeWidth={2} filter="url(#r)">
        <path fill="url(#s)" stroke="url(#t)" d="M180.882 214.82c3.301-4.237 10.089-2.173 10.472 3.186l1.174 16.433c.609 8.529-7.525 15.017-15.705 12.529-8.18-2.488-11.322-12.406-6.067-19.151l10.126-12.997Z" />
        <path fill="url(#u)" stroke="url(#v)" d="M229.469 214.82c3.301-4.237 10.089-2.173 10.472 3.186l1.174 16.433c.609 8.529-7.525 15.017-15.705 12.529-8.18-2.488-11.322-12.406-6.067-19.151l10.126-12.997Z" />
      </g>
      <defs>
        <linearGradient id="c" x1={190} x2={190} y1={-16.5} y2={218.5} gradientUnits="userSpaceOnUse">
          <stop stopColor="#fff" />
          <stop offset={1} stopColor="#fff" stopOpacity={0} />
        </linearGradient>
        <linearGradient id="d" x1={190} x2={190} y1={35} y2={196} gradientUnits="userSpaceOnUse">
          <stop stopColor="#fff" />
          <stop offset={1} stopColor="#fff" stopOpacity={0} />
        </linearGradient>
        <linearGradient id="h" x1={297} x2={297} y1={38.5} y2={164.5} gradientUnits="userSpaceOnUse">
          <stop stopColor="#fff" />
          <stop offset={1} stopColor="#fff" stopOpacity={0} />
        </linearGradient>
        <linearGradient id="i" x1={289} x2={289} y1={35} y2={151.353} gradientUnits="userSpaceOnUse">
          <stop stopColor="#fff" />
          <stop offset={1} stopColor="#fff" stopOpacity={0} />
        </linearGradient>
        <linearGradient id="l" x1={175.905} x2={161.359} y1={154.585} y2={202.404} gradientUnits="userSpaceOnUse">
          <stop stopColor="#138EFF" />
          <stop offset={1} stopColor="#00E0FF" />
        </linearGradient>
        <linearGradient id="m" x1={175.905} x2={161.359} y1={154.585} y2={202.404} gradientUnits="userSpaceOnUse">
          <stop stopColor="#fff" />
          <stop offset={1} stopColor="#fff" stopOpacity={0} />
        </linearGradient>
        <linearGradient id="n" x1={233.413} x2={218.867} y1={154.585} y2={202.404} gradientUnits="userSpaceOnUse">
          <stop stopColor="#138EFF" />
          <stop offset={1} stopColor="#00E0FF" />
        </linearGradient>
        <linearGradient id="o" x1={233.413} x2={218.867} y1={154.585} y2={202.404} gradientUnits="userSpaceOnUse">
          <stop stopColor="#fff" />
          <stop offset={1} stopColor="#fff" stopOpacity={0} />
        </linearGradient>
        <linearGradient id="p" x1={290.92} x2={276.374} y1={154.585} y2={202.404} gradientUnits="userSpaceOnUse">
          <stop stopColor="#138EFF" />
          <stop offset={1} stopColor="#00E0FF" />
        </linearGradient>
        <linearGradient id="q" x1={290.92} x2={276.374} y1={154.585} y2={202.404} gradientUnits="userSpaceOnUse">
          <stop stopColor="#fff" />
          <stop offset={1} stopColor="#fff" stopOpacity={0} />
        </linearGradient>
        <linearGradient id="s" x1={191.079} x2={176.533} y1={200.105} y2={247.925} gradientUnits="userSpaceOnUse">
          <stop stopColor="#138EFF" />
          <stop offset={1} stopColor="#00E0FF" />
        </linearGradient>
        <linearGradient id="t" x1={191.079} x2={176.533} y1={200.105} y2={247.925} gradientUnits="userSpaceOnUse">
          <stop stopColor="#fff" />
          <stop offset={1} stopColor="#fff" stopOpacity={0} />
        </linearGradient>
        <linearGradient id="u" x1={239.666} x2={225.12} y1={200.105} y2={247.925} gradientUnits="userSpaceOnUse">
          <stop stopColor="#138EFF" />
          <stop offset={1} stopColor="#00E0FF" />
        </linearGradient>
        <linearGradient id="v" x1={239.666} x2={225.12} y1={200.105} y2={247.925} gradientUnits="userSpaceOnUse">
          <stop stopColor="#fff" />
          <stop offset={1} stopColor="#fff" stopOpacity={0} />
        </linearGradient>
        <filter id="b" width={338} height={261} x={0} y={0} colorInterpolationFilters="sRGB" filterUnits="userSpaceOnUse">
          <feFlood floodOpacity={0} result="BackgroundImageFix" />
          <feColorMatrix in="SourceAlpha" result="hardAlpha" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 127 0" />
          <feOffset dx={-21} dy={15} />
          <feGaussianBlur stdDeviation={25} />
          <feComposite in2="hardAlpha" operator="out" />
          <feColorMatrix values="0 0 0 0 0.113725 0 0 0 0 0.14902 0 0 0 0 0.27451 0 0 0 0.25 0" />
          <feBlend in2="BackgroundImageFix" result="effect1_dropShadow_2_587" />
          <feBlend in="SourceGraphic" in2="effect1_dropShadow_2_587" result="shape" />
        </filter>
        <filter id="g" width={272} height={216.353} x={132} y={0} colorInterpolationFilters="sRGB" filterUnits="userSpaceOnUse">
          <feFlood floodOpacity={0} result="BackgroundImageFix" />
          <feColorMatrix in="SourceAlpha" result="hardAlpha" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 127 0" />
          <feOffset dx={-21} dy={15} />
          <feGaussianBlur stdDeviation={25} />
          <feComposite in2="hardAlpha" operator="out" />
          <feColorMatrix values="0 0 0 0 0.113438 0 0 0 0 0.148981 0 0 0 0 0.275 0 0 0 0.25 0" />
          <feBlend in2="BackgroundImageFix" result="effect1_dropShadow_2_587" />
          <feBlend in="SourceGraphic" in2="effect1_dropShadow_2_587" result="shape" />
        </filter>
        <filter id="k" width={253.399} height={149.955} x={81.002} y={118.031} colorInterpolationFilters="sRGB" filterUnits="userSpaceOnUse">
          <feFlood floodOpacity={0} result="BackgroundImageFix" />
          <feColorMatrix in="SourceAlpha" result="hardAlpha" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 127 0" />
          <feOffset dx={-21} dy={15} />
          <feGaussianBlur stdDeviation={25} />
          <feComposite in2="hardAlpha" operator="out" />
          <feColorMatrix values="0 0 0 0 0.113438 0 0 0 0 0.148981 0 0 0 0 0.275 0 0 0 0.25 0" />
          <feBlend in2="BackgroundImageFix" result="effect1_dropShadow_2_587" />
          <feColorMatrix in="SourceAlpha" result="hardAlpha" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 127 0" />
          <feOffset dy={-7} />
          <feGaussianBlur stdDeviation={20.5} />
          <feComposite in2="hardAlpha" operator="out" />
          <feColorMatrix values="0 0 0 0 0.1375 0 0 0 0 0.74125 0 0 0 0 1 0 0 0 0.3 0" />
          <feBlend in2="effect1_dropShadow_2_587" result="effect2_dropShadow_2_587" />
          <feBlend in="SourceGraphic" in2="effect2_dropShadow_2_587" result="shape" />
        </filter>
        <filter id="r" width={186.972} height={149.955} x={96.175} y={163.551} colorInterpolationFilters="sRGB" filterUnits="userSpaceOnUse">
          <feFlood floodOpacity={0} result="BackgroundImageFix" />
          <feColorMatrix in="SourceAlpha" result="hardAlpha" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 127 0" />
          <feOffset dx={-21} dy={15} />
          <feGaussianBlur stdDeviation={25} />
          <feComposite in2="hardAlpha" operator="out" />
          <feColorMatrix values="0 0 0 0 0.113438 0 0 0 0 0.148981 0 0 0 0 0.275 0 0 0 0.25 0" />
          <feBlend in2="BackgroundImageFix" result="effect1_dropShadow_2_587" />
          <feColorMatrix in="SourceAlpha" result="hardAlpha" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 127 0" />
          <feOffset dy={-7} />
          <feGaussianBlur stdDeviation={20.5} />
          <feComposite in2="hardAlpha" operator="out" />
          <feColorMatrix values="0 0 0 0 0.1375 0 0 0 0 0.74125 0 0 0 0 1 0 0 0 0.3 0" />
          <feBlend in2="effect1_dropShadow_2_587" result="effect2_dropShadow_2_587" />
          <feBlend in="SourceGraphic" in2="effect2_dropShadow_2_587" result="shape" />
        </filter>
        <clipPath id="a">
          <path
            fillRule="evenodd"
            d="M265.5 195.997c.166.002.333.003.5.003 23.748 0 43-19.252 43-43s-19.252-43-43-43c-.167 0-.334.001-.5.003V110h-11.109a66.47 66.47 0 0 0 .609-9c0-36.45-29.549-66-66-66s-66 29.55-66 66c0 3.439.263 6.817.77 10.115A43.142 43.142 0 0 0 114 110c-23.748 0-43 19.252-43 43s19.252 43 43 43c.168 0 .335-.001.502-.003l-.002.003h151v-.003Z"
            clipRule="evenodd"
          />
        </clipPath>
        <clipPath id="f" transform="translate(-132)">
          <path
            fillRule="evenodd"
            d="m343.563 151.351.361.002c17.163 0 31.076-13.913 31.076-31.076 0-17.162-13.913-31.075-31.076-31.075-.12 0-.241 0-.361.002v-.002h-8.028c.29-2.127.44-4.298.44-6.504C335.975 56.355 314.62 35 288.277 35c-26.342 0-47.697 21.355-47.697 47.698 0 2.485.19 4.926.556 7.31a31.16 31.16 0 0 0-7.06-.806c-17.163 0-31.076 13.913-31.076 31.075 0 17.163 13.913 31.076 31.076 31.076l.363-.002-.002.002h109.126v-.002Z"
            clipRule="evenodd"
          />
        </clipPath>
      </defs>
    </svg>
  ),
  heavy_rain: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width={444} height={403} fill="none">
      <foreignObject width={272} height={216.353} x={0} y={0}>
        <div
          xmlns="http://www.w3.org/1999/xhtml"
          style={{
            backdropFilter: "blur(7px)",
            clipPath: "url(#a)",
            height: "100%",
            width: "100%",
          }}
        />
      </foreignObject>
      <g data-figma-bg-blur-radius={14} filter="url(#b)">
        <mask id="e" fill="#fff">
          <path
            fillRule="evenodd"
            d="m211.563 151.351.361.002c17.163 0 31.076-13.913 31.076-31.076 0-17.162-13.913-31.075-31.076-31.075-.12 0-.241 0-.361.002v-.002h-8.028c.29-2.127.44-4.298.44-6.504C203.975 56.355 182.62 35 156.277 35c-26.342 0-47.697 21.355-47.697 47.698 0 2.485.19 4.926.556 7.31a31.16 31.16 0 0 0-7.06-.806C84.913 89.202 71 103.115 71 120.277c0 17.163 13.913 31.076 31.076 31.076l.363-.002-.002.002h109.126v-.002Z"
            clipRule="evenodd"
          />
        </mask>
        <path
          fill="url(#c)"
          fillRule="evenodd"
          d="m211.563 151.351.361.002c17.163 0 31.076-13.913 31.076-31.076 0-17.162-13.913-31.075-31.076-31.075-.12 0-.241 0-.361.002v-.002h-8.028c.29-2.127.44-4.298.44-6.504C203.975 56.355 182.62 35 156.277 35c-26.342 0-47.697 21.355-47.697 47.698 0 2.485.19 4.926.556 7.31a31.16 31.16 0 0 0-7.06-.806C84.913 89.202 71 103.115 71 120.277c0 17.163 13.913 31.076 31.076 31.076l.363-.002-.002.002h109.126v-.002Z"
          clipRule="evenodd"
          shapeRendering="crispEdges"
        />
        <path
          fill="url(#d)"
          d="m211.563 151.351.023-2a2 2 0 0 0-2.023 2h2Zm0-62.147h-2a2 2 0 0 0 2.023 2l-.023-2Zm0-.002h2a2 2 0 0 0-2-2v2Zm-8.028 0-1.982-.27a2.003 2.003 0 0 0 1.982 2.27v-2Zm-94.399.805-.452 1.948a2 2 0 0 0 2.429-2.252l-1.977.304Zm-6.697 61.344 1.569 1.239a2 2 0 0 0-1.592-3.239l.023 2Zm-.002.002-1.57-1.239a2 2 0 0 0 1.57 3.239v-2Zm109.126 0v2a2 2 0 0 0 2-2h-2Zm-.023 1.998.384.002v-4l-.338-.002-.046 4Zm.384.002c18.268 0 33.076-14.809 33.076-33.076h-4c0 16.058-13.018 29.076-29.076 29.076v4ZM245 120.277c0-18.267-14.808-33.075-33.076-33.075v4c16.058 0 29.076 13.017 29.076 29.075h4Zm-33.076-33.075c-.128 0-.256 0-.384.002l.046 4c.113-.002.225-.002.338-.002v-4Zm-2.361 2v.002h4v-.002h-4Zm-6.028 2h8.028v-4h-8.028v4Zm1.982-1.73a50.15 50.15 0 0 0 .458-6.774h-4c0 2.115-.144 4.196-.422 6.234l3.964.54Zm.458-6.774C205.975 55.25 183.724 33 156.277 33v4c25.238 0 45.698 20.46 45.698 45.698h4ZM156.277 33c-27.447 0-49.697 22.25-49.697 49.698h4C110.58 57.46 131.039 37 156.277 37v-4ZM106.58 82.698c0 2.587.198 5.13.58 7.613l3.953-.608a46.076 46.076 0 0 1-.533-7.005h-4Zm3.009 5.361a33.172 33.172 0 0 0-7.513-.857v4c2.275 0 4.487.26 6.608.754l.905-3.897Zm-7.513-.857C83.809 87.202 69 102.01 69 120.277h4c0-16.058 13.018-29.075 29.076-29.075v-4ZM69 120.277c0 18.267 14.808 33.076 33.076 33.076v-4C86.018 149.353 73 136.335 73 120.277h-4Zm33.076 33.076.385-.002-.045-4-.34.002v4Zm1.931-.761.001-.002-3.139-2.478-.002.002 3.14 2.478Zm107.556-3.239H102.437v4h109.126v-4Zm-2 1.998v.002h4v-.002h-4Z"
          mask="url(#e)"
        />
      </g>
      <foreignObject width={338} height={261} x={40} y={35}>
        <div
          xmlns="http://www.w3.org/1999/xhtml"
          style={{
            backdropFilter: "blur(7px)",
            clipPath: "url(#f)",
            height: "100%",
            width: "100%",
          }}
        />
      </foreignObject>
      <g data-figma-bg-blur-radius={14} filter="url(#g)">
        <mask id="j" fill="#fff">
          <path
            fillRule="evenodd"
            d="M305.5 230.997c.166.002.333.003.5.003 23.748 0 43-19.252 43-43s-19.252-43-43-43c-.167 0-.334.001-.5.003V145h-11.109a66.47 66.47 0 0 0 .609-9c0-36.45-29.549-66-66-66s-66 29.55-66 66c0 3.439.263 6.817.77 10.115A43.142 43.142 0 0 0 154 145c-23.748 0-43 19.252-43 43s19.252 43 43 43c.168 0 .335-.001.502-.003l-.002.003h151v-.003Z"
            clipRule="evenodd"
          />
        </mask>
        <path
          fill="url(#h)"
          fillRule="evenodd"
          d="M305.5 230.997c.166.002.333.003.5.003 23.748 0 43-19.252 43-43s-19.252-43-43-43c-.167 0-.334.001-.5.003V145h-11.109a66.47 66.47 0 0 0 .609-9c0-36.45-29.549-66-66-66s-66 29.55-66 66c0 3.439.263 6.817.77 10.115A43.142 43.142 0 0 0 154 145c-23.748 0-43 19.252-43 43s19.252 43 43 43c.168 0 .335-.001.502-.003l-.002.003h151v-.003Z"
          clipRule="evenodd"
          shapeRendering="crispEdges"
        />
        <path
          fill="url(#i)"
          d="m305.5 230.997.023-2a1.999 1.999 0 0 0-2.023 2h2Zm0-85.994h-2a2 2 0 0 0 2.023 2l-.023-2Zm0-.003h2a2 2 0 0 0-2-2v2Zm-11.109 0-1.981-.27a1.999 1.999 0 0 0 1.981 2.27v-2Zm-130.621 1.115-.452 1.948a2.001 2.001 0 0 0 2.429-2.252l-1.977.304Zm-9.268 84.882 1.57 1.24a2.001 2.001 0 0 0-1.593-3.24l.023 2Zm-.002.003-1.569-1.24A2 2 0 0 0 154.5 233v-2Zm151 0v2a2 2 0 0 0 2-2h-2Zm-.023 1.997c.174.002.348.003.523.003v-4l-.477-.003-.046 4ZM306 233c24.853 0 45-20.147 45-45h-4c0 22.644-18.356 41-41 41v4Zm45-45c0-24.853-20.147-45-45-45v4c22.644 0 41 18.356 41 41h4Zm-45-45c-.174 0-.349.001-.523.003l.046 4L306 147v-4Zm-2.5 2v.003h4V145h-4Zm-9.109 2H305.5v-4h-11.109v4Zm1.982-1.73A68.5 68.5 0 0 0 297 136h-4c0 2.962-.201 5.876-.59 8.73l3.963.54ZM297 136c0-37.555-30.445-68-68-68v4c35.346 0 64 28.654 64 64h4Zm-68-68c-37.555 0-68 30.445-68 68h4c0-35.346 28.654-64 64-64v-4Zm-68 68c0 3.542.271 7.021.793 10.419l3.954-.608A64.477 64.477 0 0 1 165 136h-4Zm3.223 8.167A45.103 45.103 0 0 0 154 143v4c3.207 0 6.326.368 9.318 1.063l.905-3.896ZM154 143c-24.853 0-45 20.147-45 45h4c0-22.644 18.356-41 41-41v-4Zm-45 45c0 24.853 20.147 45 45 45v-4c-22.644 0-41-18.356-41-41h-4Zm45 45 .525-.003-.046-4c-.159.002-.319.003-.479.003v4Zm2.069-.76.003-.003-3.139-2.48-.002.003 3.138 2.48ZM305.5 229h-151v4h151v-4Zm-2 1.997V231h4v-.003h-4Z"
          mask="url(#j)"
        />
      </g>
      <foreignObject width={241.161} height={305.265} x={202.369} y={97}>
        <div
          xmlns="http://www.w3.org/1999/xhtml"
          style={{
            backdropFilter: "blur(7px)",
            clipPath: "url(#k)",
            height: "100%",
            width: "100%",
          }}
        />
      </foreignObject>
      <g data-figma-bg-blur-radius={14} filter="url(#l)">
        <path
          fill="url(#m)"
          d="m273.617 246.541 47.814-87.5a2 2 0 0 1 1.755-1.041h47.009c1.503 0 2.469 1.596 1.771 2.928l-29.932 57.144c-.698 1.332.268 2.928 1.771 2.928h36.722c1.729 0 2.643 2.045 1.49 3.334L281.581 336.585c-1.474 1.648-4.138.038-3.364-2.033l30.774-82.352a2 2 0 0 0-1.873-2.7h-31.746c-1.519 0-2.483-1.626-1.755-2.959Z"
        />
        <path
          stroke="url(#n)"
          strokeWidth={2}
          d="M323.187 159h47.007a1 1 0 0 1 .886 1.464l-29.933 57.144c-1.045 1.998.404 4.392 2.659 4.392h36.72c.865 0 1.323 1.023.746 1.667L280.836 335.919c-.737.823-2.069.018-1.682-1.018l30.774-82.351c.732-1.961-.718-4.05-2.811-4.05h-31.745a1 1 0 0 1-.878-1.479l47.815-87.5a1 1 0 0 1 .878-.521Z"
        />
      </g>
      <foreignObject width={272} height={216.353} x={172} y={35}>
        <div
          xmlns="http://www.w3.org/1999/xhtml"
          style={{
            backdropFilter: "blur(7px)",
            clipPath: "url(#o)",
            height: "100%",
            width: "100%",
          }}
        />
      </foreignObject>
      <g data-figma-bg-blur-radius={14} filter="url(#p)">
        <mask id="s" fill="#fff">
          <path
            fillRule="evenodd"
            d="m383.563 186.351.361.002c17.163 0 31.076-13.913 31.076-31.076 0-17.162-13.913-31.075-31.076-31.075-.12 0-.241 0-.361.002v-.002h-8.028c.29-2.127.44-4.298.44-6.505C375.975 91.355 354.62 70 328.277 70c-26.342 0-47.697 21.355-47.697 47.697 0 2.486.19 4.927.556 7.31a31.166 31.166 0 0 0-7.06-.805c-17.163 0-31.076 13.913-31.076 31.075 0 17.163 13.913 31.076 31.076 31.076l.363-.002-.002.002h109.126v-.002Z"
            clipRule="evenodd"
          />
        </mask>
        <path
          fill="url(#q)"
          fillRule="evenodd"
          d="m383.563 186.351.361.002c17.163 0 31.076-13.913 31.076-31.076 0-17.162-13.913-31.075-31.076-31.075-.12 0-.241 0-.361.002v-.002h-8.028c.29-2.127.44-4.298.44-6.505C375.975 91.355 354.62 70 328.277 70c-26.342 0-47.697 21.355-47.697 47.697 0 2.486.19 4.927.556 7.31a31.166 31.166 0 0 0-7.06-.805c-17.163 0-31.076 13.913-31.076 31.075 0 17.163 13.913 31.076 31.076 31.076l.363-.002-.002.002h109.126v-.002Z"
          clipRule="evenodd"
          shapeRendering="crispEdges"
        />
        <path
          fill="url(#r)"
          d="m383.563 186.351.023-2a2 2 0 0 0-2.023 2h2Zm0-62.147h-2a2 2 0 0 0 2.023 2l-.023-2Zm0-.002h2a2 2 0 0 0-2-2v2Zm-8.028 0-1.982-.271a2.003 2.003 0 0 0 1.982 2.271v-2Zm-94.399.805-.452 1.948a1.998 1.998 0 0 0 2.429-2.252l-1.977.304Zm-6.697 61.344 1.569 1.239a2 2 0 0 0-1.592-3.239l.023 2Zm-.002.002-1.57-1.239a2 2 0 0 0 1.57 3.239v-2Zm109.126 0v2a2 2 0 0 0 2-2h-2Zm-.023 1.998.384.002v-4l-.338-.002-.046 4Zm.384.002c18.268 0 33.076-14.809 33.076-33.076h-4c0 16.058-13.018 29.076-29.076 29.076v4ZM417 155.277c0-18.267-14.808-33.075-33.076-33.075v4c16.058 0 29.076 13.017 29.076 29.075h4Zm-33.076-33.075c-.128 0-.256 0-.384.002l.046 4c.113-.002.225-.002.338-.002v-4Zm-2.361 2v.002h4v-.002h-4Zm-6.028 2h8.028v-4h-8.028v4Zm1.982-1.73c.302-2.216.458-4.478.458-6.775h-4c0 2.116-.144 4.197-.422 6.234l3.964.541Zm.458-6.775C377.975 90.25 355.724 68 328.277 68v4c25.238 0 45.698 20.46 45.698 45.697h4ZM328.277 68c-27.447 0-49.697 22.25-49.697 49.697h4C282.58 92.46 303.039 72 328.277 72v-4Zm-49.697 49.697c0 2.588.198 5.131.58 7.614l3.953-.608a46.066 46.066 0 0 1-.533-7.006h-4Zm3.009 5.362a33.183 33.183 0 0 0-7.513-.857v4c2.275 0 4.487.261 6.608.753l.905-3.896Zm-7.513-.857c-18.268 0-33.076 14.808-33.076 33.075h4c0-16.058 13.018-29.075 29.076-29.075v-4ZM241 155.277c0 18.267 14.808 33.076 33.076 33.076v-4c-16.058 0-29.076-13.018-29.076-29.076h-4Zm33.076 33.076.385-.002-.045-4-.34.002v4Zm1.931-.761.001-.002-3.139-2.478-.002.002 3.14 2.478Zm107.556-3.239H274.437v4h109.126v-4Zm-2 1.998v.002h4v-.002h-4Z"
          mask="url(#s)"
        />
      </g>
      <g strokeWidth={3} filter="url(#t)">
        <path fill="url(#u)" stroke="url(#v)" d="M117.103 244.608c3.02-3.877 9.229-1.988 9.579 2.914l1.174 16.432c.584 8.179-7.216 14.402-15.061 12.015-7.844-2.386-10.857-11.898-5.818-18.366l10.126-12.995Z" />
        <path fill="url(#w)" stroke="url(#x)" d="M174.61 244.608c3.02-3.877 9.229-1.988 9.579 2.914l1.174 16.432c.584 8.179-7.216 14.402-15.061 12.015-7.844-2.386-10.857-11.898-5.818-18.366l10.126-12.995Z" />
        <path fill="url(#y)" stroke="url(#z)" d="M232.117 244.608c3.02-3.877 9.23-1.988 9.58 2.914l1.174 16.432c.584 8.179-7.216 14.402-15.061 12.015-7.844-2.386-10.857-11.898-5.818-18.366l10.125-12.995Z" />
      </g>
      <g strokeWidth={3} filter="url(#A)">
        <path fill="url(#B)" stroke="url(#C)" d="M139.103 290.128c3.02-3.877 9.229-1.988 9.579 2.914l1.174 16.433c.584 8.178-7.216 14.401-15.061 12.015-7.844-2.387-10.857-11.898-5.818-18.366l10.126-12.996Z" />
        <path fill="url(#D)" stroke="url(#E)" d="M197.103 290.128c3.02-3.877 9.229-1.988 9.579 2.914l1.174 16.433c.584 8.178-7.216 14.401-15.061 12.015-7.844-2.387-10.857-11.898-5.818-18.366l10.126-12.996Z" />
      </g>
      <defs>
        <linearGradient id="c" x1={165} x2={165} y1={38.5} y2={164.5} gradientUnits="userSpaceOnUse">
          <stop stopColor="#fff" />
          <stop offset={1} stopColor="#fff" stopOpacity={0} />
        </linearGradient>
        <linearGradient id="d" x1={157} x2={157} y1={35} y2={151.353} gradientUnits="userSpaceOnUse">
          <stop stopColor="#fff" />
          <stop offset={1} stopColor="#fff" stopOpacity={0} />
        </linearGradient>
        <linearGradient id="h" x1={230} x2={230} y1={18.5} y2={253.5} gradientUnits="userSpaceOnUse">
          <stop stopColor="#fff" />
          <stop offset={1} stopColor="#fff" stopOpacity={0} />
        </linearGradient>
        <linearGradient id="i" x1={230} x2={230} y1={70} y2={231} gradientUnits="userSpaceOnUse">
          <stop stopColor="#fff" />
          <stop offset={1} stopColor="#fff" stopOpacity={0} />
        </linearGradient>
        <linearGradient id="m" x1={339.5} x2={314.5} y1={169} y2={277.5} gradientUnits="userSpaceOnUse">
          <stop stopColor="#FF4D00" />
          <stop offset={1} stopColor="#FFB800" />
        </linearGradient>
        <linearGradient id="n" x1={306} x2={351.5} y1={276} y2={178.5} gradientUnits="userSpaceOnUse">
          <stop stopColor="#fff" />
          <stop offset={1} stopColor="#fff" stopOpacity={0} />
        </linearGradient>
        <linearGradient id="q" x1={337} x2={337} y1={73.5} y2={199.5} gradientUnits="userSpaceOnUse">
          <stop stopColor="#fff" />
          <stop offset={1} stopColor="#fff" stopOpacity={0} />
        </linearGradient>
        <linearGradient id="r" x1={329} x2={329} y1={70} y2={186.353} gradientUnits="userSpaceOnUse">
          <stop stopColor="#fff" />
          <stop offset={1} stopColor="#fff" stopOpacity={0} />
        </linearGradient>
        <linearGradient id="u" x1={126.905} x2={112.359} y1={229.585} y2={277.404} gradientUnits="userSpaceOnUse">
          <stop stopColor="#138EFF" />
          <stop offset={1} stopColor="#00E0FF" />
        </linearGradient>
        <linearGradient id="v" x1={126.905} x2={112.359} y1={229.585} y2={277.404} gradientUnits="userSpaceOnUse">
          <stop stopColor="#fff" />
          <stop offset={1} stopColor="#fff" stopOpacity={0} />
        </linearGradient>
        <linearGradient id="w" x1={184.412} x2={169.866} y1={229.585} y2={277.404} gradientUnits="userSpaceOnUse">
          <stop stopColor="#138EFF" />
          <stop offset={1} stopColor="#00E0FF" />
        </linearGradient>
        <linearGradient id="x" x1={184.412} x2={169.866} y1={229.585} y2={277.404} gradientUnits="userSpaceOnUse">
          <stop stopColor="#fff" />
          <stop offset={1} stopColor="#fff" stopOpacity={0} />
        </linearGradient>
        <linearGradient id="y" x1={241.92} x2={227.374} y1={229.585} y2={277.404} gradientUnits="userSpaceOnUse">
          <stop stopColor="#138EFF" />
          <stop offset={1} stopColor="#00E0FF" />
        </linearGradient>
        <linearGradient id="z" x1={241.92} x2={227.374} y1={229.585} y2={277.404} gradientUnits="userSpaceOnUse">
          <stop stopColor="#fff" />
          <stop offset={1} stopColor="#fff" stopOpacity={0} />
        </linearGradient>
        <linearGradient id="B" x1={148.905} x2={134.359} y1={275.105} y2={322.925} gradientUnits="userSpaceOnUse">
          <stop stopColor="#138EFF" />
          <stop offset={1} stopColor="#00E0FF" />
        </linearGradient>
        <linearGradient id="C" x1={148.905} x2={134.359} y1={275.105} y2={322.925} gradientUnits="userSpaceOnUse">
          <stop stopColor="#fff" />
          <stop offset={1} stopColor="#fff" stopOpacity={0} />
        </linearGradient>
        <linearGradient id="D" x1={206.905} x2={192.359} y1={275.105} y2={322.925} gradientUnits="userSpaceOnUse">
          <stop stopColor="#138EFF" />
          <stop offset={1} stopColor="#00E0FF" />
        </linearGradient>
        <linearGradient id="E" x1={206.905} x2={192.359} y1={275.105} y2={322.925} gradientUnits="userSpaceOnUse">
          <stop stopColor="#fff" />
          <stop offset={1} stopColor="#fff" stopOpacity={0} />
        </linearGradient>
        <filter id="b" width={272} height={216.353} x={0} y={0} colorInterpolationFilters="sRGB" filterUnits="userSpaceOnUse">
          <feFlood floodOpacity={0} result="BackgroundImageFix" />
          <feColorMatrix in="SourceAlpha" result="hardAlpha" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 127 0" />
          <feOffset dx={-21} dy={15} />
          <feGaussianBlur stdDeviation={25} />
          <feComposite in2="hardAlpha" operator="out" />
          <feColorMatrix values="0 0 0 0 0.113438 0 0 0 0 0.148981 0 0 0 0 0.275 0 0 0 0.25 0" />
          <feBlend in2="BackgroundImageFix" result="effect1_dropShadow_2_766" />
          <feBlend in="SourceGraphic" in2="effect1_dropShadow_2_766" result="shape" />
        </filter>
        <filter id="g" width={338} height={261} x={40} y={35} colorInterpolationFilters="sRGB" filterUnits="userSpaceOnUse">
          <feFlood floodOpacity={0} result="BackgroundImageFix" />
          <feColorMatrix in="SourceAlpha" result="hardAlpha" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 127 0" />
          <feOffset dx={-21} dy={15} />
          <feGaussianBlur stdDeviation={25} />
          <feComposite in2="hardAlpha" operator="out" />
          <feColorMatrix values="0 0 0 0 0.113725 0 0 0 0 0.14902 0 0 0 0 0.27451 0 0 0 0.25 0" />
          <feBlend in2="BackgroundImageFix" result="effect1_dropShadow_2_766" />
          <feBlend in="SourceGraphic" in2="effect1_dropShadow_2_766" result="shape" />
        </filter>
        <filter id="l" width={241.161} height={305.265} x={202.369} y={97} colorInterpolationFilters="sRGB" filterUnits="userSpaceOnUse">
          <feFlood floodOpacity={0} result="BackgroundImageFix" />
          <feColorMatrix in="SourceAlpha" result="hardAlpha" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 127 0" />
          <feOffset dx={-21} dy={15} />
          <feGaussianBlur stdDeviation={25} />
          <feComposite in2="hardAlpha" operator="out" />
          <feColorMatrix values="0 0 0 0 0.113438 0 0 0 0 0.148981 0 0 0 0 0.275 0 0 0 0.25 0" />
          <feBlend in2="BackgroundImageFix" result="effect1_dropShadow_2_766" />
          <feColorMatrix in="SourceAlpha" result="hardAlpha" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 127 0" />
          <feOffset />
          <feGaussianBlur stdDeviation={30.5} />
          <feComposite in2="hardAlpha" operator="out" />
          <feColorMatrix values="0 0 0 0 1 0 0 0 0 0.721569 0 0 0 0 0.00392157 0 0 0 0.49 0" />
          <feBlend in2="effect1_dropShadow_2_766" result="effect2_dropShadow_2_766" />
          <feBlend in="SourceGraphic" in2="effect2_dropShadow_2_766" result="shape" />
        </filter>
        <filter id="p" width={272} height={216.353} x={172} y={35} colorInterpolationFilters="sRGB" filterUnits="userSpaceOnUse">
          <feFlood floodOpacity={0} result="BackgroundImageFix" />
          <feColorMatrix in="SourceAlpha" result="hardAlpha" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 127 0" />
          <feOffset dx={-21} dy={15} />
          <feGaussianBlur stdDeviation={25} />
          <feComposite in2="hardAlpha" operator="out" />
          <feColorMatrix values="0 0 0 0 0.113438 0 0 0 0 0.148981 0 0 0 0 0.275 0 0 0 0.25 0" />
          <feBlend in2="BackgroundImageFix" result="effect1_dropShadow_2_766" />
          <feBlend in="SourceGraphic" in2="effect1_dropShadow_2_766" result="shape" />
        </filter>
        <filter id="t" width={253.399} height={149.955} x={32.002} y={193.031} colorInterpolationFilters="sRGB" filterUnits="userSpaceOnUse">
          <feFlood floodOpacity={0} result="BackgroundImageFix" />
          <feColorMatrix in="SourceAlpha" result="hardAlpha" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 127 0" />
          <feOffset dx={-21} dy={15} />
          <feGaussianBlur stdDeviation={25} />
          <feComposite in2="hardAlpha" operator="out" />
          <feColorMatrix values="0 0 0 0 0.113438 0 0 0 0 0.148981 0 0 0 0 0.275 0 0 0 0.25 0" />
          <feBlend in2="BackgroundImageFix" result="effect1_dropShadow_2_766" />
          <feColorMatrix in="SourceAlpha" result="hardAlpha" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 127 0" />
          <feOffset dy={-7} />
          <feGaussianBlur stdDeviation={20.5} />
          <feComposite in2="hardAlpha" operator="out" />
          <feColorMatrix values="0 0 0 0 0.1375 0 0 0 0 0.74125 0 0 0 0 1 0 0 0 0.3 0" />
          <feBlend in2="effect1_dropShadow_2_766" result="effect2_dropShadow_2_766" />
          <feBlend in="SourceGraphic" in2="effect2_dropShadow_2_766" result="shape" />
        </filter>
        <filter id="A" width={196.385} height={149.955} x={54.002} y={238.551} colorInterpolationFilters="sRGB" filterUnits="userSpaceOnUse">
          <feFlood floodOpacity={0} result="BackgroundImageFix" />
          <feColorMatrix in="SourceAlpha" result="hardAlpha" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 127 0" />
          <feOffset dx={-21} dy={15} />
          <feGaussianBlur stdDeviation={25} />
          <feComposite in2="hardAlpha" operator="out" />
          <feColorMatrix values="0 0 0 0 0.113438 0 0 0 0 0.148981 0 0 0 0 0.275 0 0 0 0.25 0" />
          <feBlend in2="BackgroundImageFix" result="effect1_dropShadow_2_766" />
          <feColorMatrix in="SourceAlpha" result="hardAlpha" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 127 0" />
          <feOffset dy={-7} />
          <feGaussianBlur stdDeviation={20.5} />
          <feComposite in2="hardAlpha" operator="out" />
          <feColorMatrix values="0 0 0 0 0.1375 0 0 0 0 0.74125 0 0 0 0 1 0 0 0 0.3 0" />
          <feBlend in2="effect1_dropShadow_2_766" result="effect2_dropShadow_2_766" />
          <feBlend in="SourceGraphic" in2="effect2_dropShadow_2_766" result="shape" />
        </filter>
        <clipPath id="a">
          <path
            fillRule="evenodd"
            d="m211.563 151.351.361.002c17.163 0 31.076-13.913 31.076-31.076 0-17.162-13.913-31.075-31.076-31.075-.12 0-.241 0-.361.002v-.002h-8.028c.29-2.127.44-4.298.44-6.504C203.975 56.355 182.62 35 156.277 35c-26.342 0-47.697 21.355-47.697 47.698 0 2.485.19 4.926.556 7.31a31.16 31.16 0 0 0-7.06-.806C84.913 89.202 71 103.115 71 120.277c0 17.163 13.913 31.076 31.076 31.076l.363-.002-.002.002h109.126v-.002Z"
            clipRule="evenodd"
          />
        </clipPath>
        <clipPath id="f" transform="translate(-40 -35)">
          <path
            fillRule="evenodd"
            d="M305.5 230.997c.166.002.333.003.5.003 23.748 0 43-19.252 43-43s-19.252-43-43-43c-.167 0-.334.001-.5.003V145h-11.109a66.47 66.47 0 0 0 .609-9c0-36.45-29.549-66-66-66s-66 29.55-66 66c0 3.439.263 6.817.77 10.115A43.142 43.142 0 0 0 154 145c-23.748 0-43 19.252-43 43s19.252 43 43 43c.168 0 .335-.001.502-.003l-.002.003h151v-.003Z"
            clipRule="evenodd"
          />
        </clipPath>
        <clipPath id="k" transform="translate(-202.369 -97)">
          <path d="m273.617 246.541 47.814-87.5a2 2 0 0 1 1.755-1.041h47.009c1.503 0 2.469 1.596 1.771 2.928l-29.932 57.144c-.698 1.332.268 2.928 1.771 2.928h36.722c1.729 0 2.643 2.045 1.49 3.334L281.581 336.585c-1.474 1.648-4.138.038-3.364-2.033l30.774-82.352a2 2 0 0 0-1.873-2.7h-31.746c-1.519 0-2.483-1.626-1.755-2.959Z" />
        </clipPath>
        <clipPath id="o" transform="translate(-172 -35)">
          <path
            fillRule="evenodd"
            d="m383.563 186.351.361.002c17.163 0 31.076-13.913 31.076-31.076 0-17.162-13.913-31.075-31.076-31.075-.12 0-.241 0-.361.002v-.002h-8.028c.29-2.127.44-4.298.44-6.505C375.975 91.355 354.62 70 328.277 70c-26.342 0-47.697 21.355-47.697 47.697 0 2.486.19 4.927.556 7.31a31.166 31.166 0 0 0-7.06-.805c-17.163 0-31.076 13.913-31.076 31.075 0 17.163 13.913 31.076 31.076 31.076l.363-.002-.002.002h109.126v-.002Z"
            clipRule="evenodd"
          />
        </clipPath>
      </defs>
    </svg>
  ),
  clear: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width={354} height={354} fill="none">
      <g filter="url(#a)">
        <circle cx={176.62} cy={176.62} r={76.62} fill="#FFC701" fillOpacity={0.35} />
      </g>
      <circle cx={176.621} cy={179.132} r={73.108} fill="url(#b)" stroke="url(#c)" strokeWidth={2} />
      <mask
        id="e"
        width={149}
        height={108}
        x={102}
        y={146}
        maskUnits="userSpaceOnUse"
        style={{
          maskType: "alpha",
        }}
      >
        <path fill="url(#d)" d="M250.729 179.133c0 40.928-33.18 74.108-74.108 74.108-40.929 0-74.109-33.18-74.109-74.108 0-40.929 27.527-31.402 68.456-31.402s79.761-9.527 79.761 31.402Z" />
      </mask>
      <g filter="url(#f)" mask="url(#e)">
        <path
          fill="#E18700"
          fillRule="evenodd"
          d="M289.667 314.788c0 29.83-24.182 54.011-54.011 54.011-.21 0-.419-.001-.628-.003v.003H45.361l.002-.003c-.21.003-.42.004-.63.004-29.83 0-54.011-24.182-54.011-54.011 0-29.83 24.181-54.011 54.01-54.011 25.713 0 47.229 17.966 52.678 42.03l18.291-23.19 57.151-18.841h62.176v.004c.209-.003.418-.004.628-.004 29.829 0 54.011 24.182 54.011 54.011Z"
          clipRule="evenodd"
        />
      </g>
      <defs>
        <linearGradient id="b" x1={176.621} x2={176.621} y1={105.024} y2={253.241} gradientUnits="userSpaceOnUse">
          <stop stopColor="#FFE600" />
          <stop offset={1} stopColor="#FF7A00" />
        </linearGradient>
        <linearGradient id="c" x1={176.621} x2={176.621} y1={105.024} y2={253.241} gradientUnits="userSpaceOnUse">
          <stop stopColor="#fff" />
          <stop offset={1} stopColor="#fff" stopOpacity={0} />
        </linearGradient>
        <linearGradient id="d" x1={176.621} x2={176.621} y1={105.024} y2={253.241} gradientUnits="userSpaceOnUse">
          <stop stopColor="#FFD600" />
          <stop offset={1} stopColor="#FF7A00" />
        </linearGradient>
        <filter id="a" width={353.241} height={353.241} x={0} y={0} colorInterpolationFilters="sRGB" filterUnits="userSpaceOnUse">
          <feFlood floodOpacity={0} result="BackgroundImageFix" />
          <feBlend in="SourceGraphic" in2="BackgroundImageFix" result="shape" />
          <feGaussianBlur result="effect1_foregroundBlur_2_698" stdDeviation={50} />
        </filter>
        <filter id="f" width={326.945} height={136.023} x={-23.278} y={246.777} colorInterpolationFilters="sRGB" filterUnits="userSpaceOnUse">
          <feFlood floodOpacity={0} result="BackgroundImageFix" />
          <feBlend in="SourceGraphic" in2="BackgroundImageFix" result="shape" />
          <feGaussianBlur result="effect1_foregroundBlur_2_698" stdDeviation={7} />
        </filter>
      </defs>
    </svg>
  ),
  twitter: Twitter,
  check: Check,
};
