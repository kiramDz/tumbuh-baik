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
  twitter: Twitter,
  check: Check,
};
